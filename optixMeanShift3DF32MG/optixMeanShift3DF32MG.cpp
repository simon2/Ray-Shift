//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

// Multi-GPU (C4) variant of ../optixMeanShift: seeds are sharded across all
// visible GPUs; every GPU builds its own copy of the disk GAS over the full
// point set and runs the (B2/B3/C2-optimized) mean-shift trajectories of its
// contiguous seed range. Trajectories are independent and outputs are written
// by ORIGINAL seed id, so the resulting clusters.csv is byte-identical to the
// single-GPU sample for any GPU count.
//
// Timing: per-GPU setup (upload/AABB/build/compact) runs concurrently and is
// reported per GPU; the PHASE line uses bvh = max over GPUs and search = wall
// clock of the launch phase (all GPUs released together after a barrier).
//
// Run: ./optixMeanShiftMG --sample all --bandwidth 1500 --all-seeds [--gpus N]

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include "optixMeanShift3DF32MG.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <array>

#define MAX_ITER 300

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

extern "C" void launch_fill_aabbs( OptixAabb* d_aabb, const float4* d_p,
                                   int n, float half_extent, cudaStream_t stream );
extern "C" void launch_morton_reorder( float4* d_p, int n, float maxr, cudaStream_t stream );


// ---------------------------------------------------------------------------
// Host helpers (identical to ../optixMeanShift)
// ---------------------------------------------------------------------------

struct Args {
    std::string input = "../data/chicago_crime.data";
    std::string output = "clusters.csv";
    std::string sample = "10000";
    double bandwidth = -1.0;   // <0 => estimate
    double quantile = 0.2;
    unsigned seed = 42;
    bool all_seeds = false;
    int max_iter = MAX_ITER;
    int chunk = 0;             // C2: iterations per launch; 0 = max_iter (single launch)
    int morton = 1;            // B6: Morton-reorder each GPU's point copy (byte-exact)
    std::string shard_mode = "contiguous";  // cross-GPU seed assignment: contiguous | strided
    int gpus = 0;              // C4: GPUs to use; 0 = all visible
    int seedsort = -1;         // C5: Morton-order each shard's seed slots (byte-exact). -1=auto, 0=off, 1=on
    int shard_block = 256;     // block size B for --shard blockstrided (Morton + block round-robin)
};

// C5: below this many seeds PER CARD the launch underfills the GPU and sorting
// concentrates dense seeds into few warps (load imbalance) -- a net loss.
// Crossover measured at ~32k seeds on an RTX A6000 (84 SMs, single-card sweep).
#define SEEDSORT_AUTO_MIN 32768

static Args parse_args( int argc, char** argv )
{
    Args a;
    for( int i = 1; i < argc; ++i )
    {
        std::string k = argv[i];
        auto next = [&]( const char* name ) -> std::string {
            if( i + 1 >= argc ) { std::fprintf( stderr, "Missing value for %s\n", name ); std::exit( 2 ); }
            return argv[++i];
        };
        if( k == "--input" ) a.input = next( "--input" );
        else if( k == "--output" ) a.output = next( "--output" );
        else if( k == "--sample" ) a.sample = next( "--sample" );
        else if( k == "--bandwidth" ) a.bandwidth = std::stod( next( "--bandwidth" ) );
        else if( k == "--quantile" ) a.quantile = std::stod( next( "--quantile" ) );
        else if( k == "--seed" ) a.seed = (unsigned)std::stoul( next( "--seed" ) );
        else if( k == "--all-seeds" ) a.all_seeds = true;
        else if( k == "--max-iter" ) a.max_iter = std::stoi( next( "--max-iter" ) );
        else if( k == "--chunk" ) a.chunk = std::stoi( next( "--chunk" ) );
        else if( k == "--gpus" ) a.gpus = std::stoi( next( "--gpus" ) );
        else if( k == "--morton" ) a.morton = std::stoi( next( "--morton" ) );
        else if( k == "--shard" ) a.shard_mode = next( "--shard" );
        else if( k == "--shard-block" ) a.shard_block = std::stoi( next( "--shard-block" ) );
        else if( k == "--seedsort" ) a.seedsort = std::stoi( next( "--seedsort" ) );
        else if( k == "-h" || k == "--help" )
        {
            std::printf( "Usage: %s [--input f] [--output f] [--sample N|all] "
                         "[--bandwidth ft] [--quantile q] [--seed s] [--all-seeds] "
                         "[--max-iter N] [--chunk N] [--gpus N]\n", argv[0] );
            std::exit( 0 );
        }
        else { std::fprintf( stderr, "Unknown argument: %s\n", k.c_str() ); std::exit( 2 ); }
    }
    return a;
}

static bool parse_double( const std::string& s, double& out )
{
    if( s.empty() ) return false;
    try { size_t pos = 0; out = std::stod( s, &pos ); return pos == s.size(); }
    catch( ... ) { return false; }
}

static void load_xy( const std::string& path, std::vector<std::string>& cases,
                     std::vector<double>& xs, std::vector<double>& ys, std::vector<double>& zs )
{
    std::ifstream in( path );
    if( !in ) { std::fprintf( stderr, "Could not open input: %s\n", path.c_str() ); std::exit( 1 ); }
    std::string line;
    if( !std::getline( in, line ) ) { std::fprintf( stderr, "Empty input file\n" ); std::exit( 1 ); }

    int ci_case = -1, ci_x = -1, ci_y = -1, ci_z = -1, idx = 0;
    {
        std::stringstream hs( line );
        std::string col;
        while( std::getline( hs, col, ',' ) )
        {
            if( col == "CASE#" ) ci_case = idx;
            else if( col == "X COORDINATE" ) ci_x = idx;
            else if( col == "Y COORDINATE" ) ci_y = idx;
            else if( col == "Z COORDINATE" ) ci_z = idx;
            ++idx;
        }
    }
    if( ci_case < 0 || ci_x < 0 || ci_y < 0 || ci_z < 0 )
    { std::fprintf( stderr, "Header missing CASE#/X/Y/Z COORDINATE columns\n" ); std::exit( 1 ); }
    const int need = std::max( std::max( ci_case, ci_z ), std::max( ci_x, ci_y ) );

    long before = 0;
    std::vector<std::string> f;
    f.reserve( 8 );
    while( std::getline( in, line ) )
    {
        ++before;
        f.clear();
        std::stringstream ls( line );
        std::string cell;
        while( std::getline( ls, cell, ',' ) ) f.push_back( cell );
        if( (int)f.size() <= need ) continue;
        double x, y, z;
        if( !parse_double( f[ci_x], x ) || !parse_double( f[ci_y], y ) || !parse_double( f[ci_z], z ) ) continue;
        cases.push_back( f[ci_case] );
        xs.push_back( x );
        ys.push_back( y );
        zs.push_back( z );
    }
    std::printf( "Loaded %ld rows, %zu valid after dropping missing coordinates.\n", before, xs.size() );
}

static double estimate_bandwidth( const std::vector<double>& xs, const std::vector<double>& ys,
                                  const std::vector<double>& zs, double quantile, unsigned seed )
{
    const int n = (int)xs.size();
    const int n_sub = std::min( 500, n );
    int k = (int)( n_sub * quantile );
    if( k < 1 ) k = 1;
    if( k > n_sub ) k = n_sub;

    std::mt19937 rng( seed );
    std::vector<int> idx( n );
    for( int i = 0; i < n; ++i ) idx[i] = i;
    std::shuffle( idx.begin(), idx.end(), rng );

    std::vector<double> d2( n_sub );
    double acc = 0.0;
    for( int a = 0; a < n_sub; ++a )
    {
        const double x = xs[idx[a]], y = ys[idx[a]], z = zs[idx[a]];
        for( int b = 0; b < n_sub; ++b )
        {
            const double dx = xs[idx[b]] - x, dy = ys[idx[b]] - y, dz = zs[idx[b]] - z;
            d2[b] = dx * dx + dy * dy + dz * dz;
        }
        std::nth_element( d2.begin(), d2.begin() + ( k - 1 ), d2.end() );
        acc += std::sqrt( d2[k - 1] );
    }
    return acc / (double)n_sub;
}

static void bin_seeds( const std::vector<double>& xs, const std::vector<double>& ys,
                       const std::vector<double>& zs, double bw,
                       std::vector<double>& sx, std::vector<double>& sy, std::vector<double>& sz )
{
    std::unordered_map<long long, int> bins;
    std::unordered_map<long long, std::array<int, 3>> coords;
    bins.reserve( xs.size() );
    for( size_t i = 0; i < xs.size(); ++i )
    {
        const long long bx = (long long)std::llround( xs[i] / bw );
        const long long by = (long long)std::llround( ys[i] / bw );
        const long long bz = (long long)std::llround( zs[i] / bw );
        const long long key = ( bx << 42 ) ^ ( by << 21 ) ^ ( bz & 0x1fffffLL );
        if( bins.find( key ) == bins.end() )
            coords[key] = { (int)bx, (int)by, (int)bz };
        bins[key]++;
    }
    for( const auto& kv : bins )
    {
        const auto& c = coords[kv.first];
        sx.push_back( c[0] * bw );
        sy.push_back( c[1] * bw );
        sz.push_back( c[2] * bw );
    }
}

static void dedup_centers( const std::vector<double>& cx, const std::vector<double>& cy,
                           const std::vector<double>& cz, const std::vector<int>& dens, double bw,
                           std::vector<double>& ux, std::vector<double>& uy, std::vector<double>& uz )
{
    const int m = (int)cx.size();
    std::vector<int> order( m );
    for( int i = 0; i < m; ++i ) order[i] = i;
    std::stable_sort( order.begin(), order.end(), [&]( int a, int b ) { return dens[a] > dens[b]; } );
    const double bw2 = bw * bw;
    std::vector<char> unique( m, 1 );
    for( int oi = 0; oi < m; ++oi )
    {
        const int i = order[oi];
        if( !unique[i] ) continue;
        for( int oj = 0; oj < m; ++oj )
        {
            const int j = order[oj];
            if( j == i || !unique[j] ) continue;
            const double dx = cx[i] - cx[j], dy = cy[i] - cy[j], dz = cz[i] - cz[j];
            if( dx * dx + dy * dy + dz * dz <= bw2 ) unique[j] = 0;
        }
        unique[i] = 1;
    }
    for( int i = 0; i < m; ++i )
        if( unique[i] ) { ux.push_back( cx[i] ); uy.push_back( cy[i] ); uz.push_back( cz[i] ); }
}

// One label per point: nearest surviving center (host; O(N*K), K small).
static void assign_labels( const std::vector<double>& xs, const std::vector<double>& ys,
                           const std::vector<double>& zs,
                           const std::vector<double>& ux, const std::vector<double>& uy,
                           const std::vector<double>& uz, std::vector<int>& labels )
{
    const int n = (int)xs.size(), k = (int)ux.size();
    labels.assign( n, 0 );
    for( int i = 0; i < n; ++i )
    {
        double best = 1e308; int bl = 0;
        for( int c = 0; c < k; ++c )
        {
            const double dx = xs[i] - ux[c], dy = ys[i] - uy[c], dz = zs[i] - uz[c];
            const double d = dx * dx + dy * dy + dz * dz;
            if( d < best ) { best = d; bl = c; }
        }
        labels[i] = bl;
    }
}


static void context_log_cb( unsigned int level, const char* tag, const char* message, void* )
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}


// C5: 2D Morton code of a point already normalized to [0,1]^2, 21 bits/axis.
// Only orders seed slots, so the grid resolution has no effect on results.
static unsigned long long morton3d_unit( float sx, float sy, float sz )
{
    auto part1by2 = []( unsigned int x ) -> unsigned long long {
        unsigned long long v = x & 0x1FFFFF;
        v = ( v | ( v << 32 ) ) & 0x001F00000000FFFFull;
        v = ( v | ( v << 16 ) ) & 0x001F0000FF0000FFull;
        v = ( v | ( v << 8 ) )  & 0x100F00F00F00F00Full;
        v = ( v | ( v << 4 ) )  & 0x10C30C30C30C30C3ull;
        v = ( v | ( v << 2 ) )  & 0x1249249249249249ull;
        return v;
    };
    sx = std::min( std::max( sx, 0.f ), 1.f );
    sy = std::min( std::max( sy, 0.f ), 1.f );
    sz = std::min( std::max( sz, 0.f ), 1.f );
    return ( part1by2( (unsigned int)( sx * 2097151.f ) ) << 2 )
         | ( part1by2( (unsigned int)( sy * 2097151.f ) ) << 1 )
         |   part1by2( (unsigned int)( sz * 2097151.f ) );
}


// Per-GPU timing report, filled by the worker threads.
struct GpuTimes
{
    double upload = 0.0, reorder = 0.0, aabb = 0.0, build = 0.0, compact = 0.0;
    double seedsort = 0.0;
    float  search = 0.f;
    int    nseeds = 0;
    int    rounds = 0;
};


// ---------------------------------------------------------------------------

int main( int argc, char** argv )
{
    Args args = parse_args( argc, argv );

    std::vector<std::string> cases;
    std::vector<double> xs, ys, zs;
    load_xy( args.input, cases, xs, ys, zs );

    bool take_all = ( args.sample == "all" || args.sample == "0" );
    if( !take_all )
    {
        long want = std::stol( args.sample );
        if( want > 0 && want < (long)xs.size() )
        {
            std::mt19937 rng( args.seed );
            std::vector<int> idx( xs.size() );
            for( size_t i = 0; i < idx.size(); ++i ) idx[i] = (int)i;
            std::shuffle( idx.begin(), idx.end(), rng );
            idx.resize( want );
            std::vector<std::string> c2; std::vector<double> x2, y2;
            c2.reserve( want ); x2.reserve( want ); y2.reserve( want );
            for( int id : idx ) { c2.push_back( cases[id] ); x2.push_back( xs[id] ); y2.push_back( ys[id] ); }
            cases.swap( c2 ); xs.swap( x2 ); ys.swap( y2 );
            std::printf( "Sampled %ld of original points (seed=%u).\n", want, args.seed );
        }
    }
    const int n = (int)xs.size();
    std::printf( "Clustering %d points on X/Y/Z coordinates (feet).\n", n );
    if( n == 0 ) { std::fprintf( stderr, "No points to cluster.\n" ); return 1; }

    using clk = std::chrono::steady_clock;
    auto ms_since = []( clk::time_point a, clk::time_point b ) {
        return std::chrono::duration<double, std::milli>( b - a ).count();
    };
    double t_estimate = 0.0, t_binning = 0.0, t_dedup = 0.0;

    double bw = args.bandwidth;
    if( bw <= 0.0 )
    {
        auto e0 = clk::now();
        bw = estimate_bandwidth( xs, ys, zs, args.quantile, args.seed );
        t_estimate = ms_since( e0, clk::now() );
        std::printf( "Estimated bandwidth: %.2f feet (quantile=%.2f).\n", bw, args.quantile );
    }
    else std::printf( "Using bandwidth: %.2f feet.\n", bw );
    if( bw <= 0.0 ) { std::fprintf( stderr, "Bandwidth is non-positive.\n" ); return 1; }

    // ---- seeds ----
    std::vector<double> sx, sy, szv;
    if( args.all_seeds )
    {
        sx = xs; sy = ys; szv = zs;
        std::printf( "All-seeds: %d seeds (one per point), no bin seeding.\n", n );
    }
    else
    {
        auto b0 = clk::now();
        bin_seeds( xs, ys, zs, bw, sx, sy, szv );
        t_binning = ms_since( b0, clk::now() );
        std::printf( "Bin seeding: %zu seeds from %d points.\n", sx.size(), n );
    }
    const int nseeds = (int)sx.size();

    // ---- recenter ----
    const double offx = *std::min_element( xs.begin(), xs.end() );
    const double offy = *std::min_element( ys.begin(), ys.end() );
    const double offz = *std::min_element( zs.begin(), zs.end() );
    const double maxr = std::max( { *std::max_element( xs.begin(), xs.end() ) - offx,
                                    *std::max_element( ys.begin(), ys.end() ) - offy,
                                    *std::max_element( zs.begin(), zs.end() ) - offz } );

    std::vector<float4> hp( n );
    for( int i = 0; i < n; ++i )
        hp[i] = make_float4( (float)( xs[i] - offx ), (float)( ys[i] - offy ),
                             (float)( zs[i] - offz ), 0.f );
    std::vector<float4> hseeds( nseeds );
    for( int i = 0; i < nseeds; ++i )
        hseeds[i] = make_float4( (float)( sx[i] - offx ), (float)( sy[i] - offy ),
                                 (float)( szv[i] - offz ), 0.f );

    const double stop_thresh = 1e-3 * bw;

    try
    {
        CUDA_CHECK( cudaFree( 0 ) );
        OPTIX_CHECK( optixInit() );

        int avail = 0;
        CUDA_CHECK( cudaGetDeviceCount( &avail ) );
        int G = args.gpus > 0 ? std::min( args.gpus, avail ) : avail;
        G = std::max( 1, std::min( G, nseeds ) );
        std::printf( "Using %d of %d visible GPUs.\n", G, avail );

        // --shard strided: permute the seed order round-robin across GPUs so
        // each contiguous shard is a spatially mixed sample (defeats hot-region
        // imbalance on coherent file orders, e.g. DIMACS usa). Outputs are
        // scattered back through the permutation -> byte-identical.
        std::vector<int> sperm;
        if( args.shard_mode == "strided" && G > 1 )
        {
            sperm.resize( nseeds );
            int kk = 0;
            for( int g = 0; g < G; ++g )
                for( int i = g; i < nseeds; i += G ) sperm[kk++] = i;
            std::vector<float4> tmp( nseeds );
            for( int j = 0; j < nseeds; ++j ) tmp[j] = hseeds[sperm[j]];
            hseeds.swap( tmp );
            std::printf( "Seed shard: strided (round-robin across %d GPUs).\n", G );
        }
        else if( args.shard_mode == "blockstrided" && G > 1 )
        {
            // Morton + block round-robin: globally 3D-Morton-sort the seeds, then
            // deal blocks of B (Morton-consecutive) round-robin across GPUs.
            // Dense Morton sweep WITHIN each block (locality) + blocks spread
            // across space (balance). Outputs scattered back via sperm.
            const float inv = maxr > 0.0 ? 1.f / (float)maxr : 0.f;
            std::vector<unsigned long long> mkey( nseeds );
            for( int i = 0; i < nseeds; ++i )
                mkey[i] = morton3d_unit( hseeds[i].x * inv, hseeds[i].y * inv, hseeds[i].z * inv );
            std::vector<int> mord( nseeds );
            for( int i = 0; i < nseeds; ++i ) mord[i] = i;
            std::sort( mord.begin(), mord.end(),
                       [&]( int a, int b ){ return mkey[a] < mkey[b]; } );
            const int B = args.shard_block > 0 ? args.shard_block : 256;
            const int nblk = ( nseeds + B - 1 ) / B;
            sperm.resize( nseeds );
            int kk = 0;
            for( int g = 0; g < G; ++g )
                for( int blk = g; blk < nblk; blk += G )
                {
                    const int s = blk * B, e = std::min( s + B, nseeds );
                    for( int p = s; p < e; ++p ) sperm[kk++] = mord[p];
                }
            std::vector<float4> tmp( nseeds );
            for( int j = 0; j < nseeds; ++j ) tmp[j] = hseeds[sperm[j]];
            hseeds.swap( tmp );
            std::printf( "Seed shard: block-strided (Morton, block=%d, %d GPUs).\n", B, G );
        }

        // Contiguous seed shards.
        std::vector<int> shard( G + 1 );
        for( int g = 0; g <= G; ++g ) shard[g] = (int)( (long long)nseeds * g / G );

        // OptiX-IR loaded once, shared read-only by all worker threads.
        size_t      irSize = 0;
        const char* ir = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixMeanShift3DF32MG.cu", irSize );

        // Shared outputs, indexed by original seed id.
        std::vector<double> ocx( nseeds ), ocy( nseeds ), ocz( nseeds );
        std::vector<int>    ocount( nseeds ), oconv( nseeds );

        std::vector<GpuTimes>    times( G );
        std::vector<std::string> errs( G );

        std::mutex              mtx;
        std::condition_variable cv_ready, cv_go;
        int  ready = 0;
        bool go    = false;

        // No AABB pad (same decision as ../optixMeanShift): with integer
        // coordinates and an integer bandwidth the box endpoints x +/- bw are
        // exact in float, so the hardware cull's per-axis comparisons are
        // exact. maxr stays: the Morton quantizer (B6) still needs it.
        const float bwf       = (float)bw;
        const int   chunk_cfg = args.chunk > 0 ? args.chunk : args.max_iter;

        auto worker = [&]( int g )
        {
            bool counted = false;
            try
            {
                CUDA_CHECK( cudaSetDevice( g ) );
                CUDA_CHECK( cudaFree( 0 ) );

                OptixDeviceContext context = nullptr;
                {
                    OptixDeviceContextOptions options = {};
                    options.logCallbackFunction = &context_log_cb;
                    options.logCallbackLevel    = 2;   // errors only: 7 GPUs would spam
                    OPTIX_CHECK( optixDeviceContextCreate( 0, &options, &context ) );
                }

                const int s0 = shard[g], ns = shard[g + 1] - shard[g];
                times[g].nseeds = ns;

                // Points upload (timed).
                float4* d_p = nullptr;
                {
                    cudaEvent_t p0, p1;
                    CUDA_CHECK( cudaEventCreate( &p0 ) ); CUDA_CHECK( cudaEventCreate( &p1 ) );
                    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_p ), n * sizeof( float4 ) ) );
                    CUDA_CHECK( cudaEventRecord( p0 ) );
                    CUDA_CHECK( cudaMemcpy( d_p, hp.data(), n * sizeof( float4 ), cudaMemcpyHostToDevice ) );
                    CUDA_CHECK( cudaEventRecord( p1 ) );
                    CUDA_CHECK( cudaEventSynchronize( p1 ) );
                    float f = 0.f; CUDA_CHECK( cudaEventElapsedTime( &f, p0, p1 ) ); times[g].upload = f;
                    cudaEventDestroy( p0 ); cudaEventDestroy( p1 );
                }

                // B6: Morton-reorder this GPU's own point copy (byte-exact).
                if( args.morton )
                {
                    cudaEvent_t r0, r1;
                    CUDA_CHECK( cudaEventCreate( &r0 ) ); CUDA_CHECK( cudaEventCreate( &r1 ) );
                    CUDA_CHECK( cudaEventRecord( r0 ) );
                    launch_morton_reorder( d_p, n, (float)maxr, 0 );
                    CUDA_CHECK( cudaEventRecord( r1 ) );
                    CUDA_CHECK( cudaEventSynchronize( r1 ) );
                    float fr = 0.f; CUDA_CHECK( cudaEventElapsedTime( &fr, r0, r1 ) ); times[g].reorder = fr;
                    cudaEventDestroy( r0 ); cudaEventDestroy( r1 );
                }

                // GAS (identical to the single-GPU sample).
                OptixTraversableHandle gas_handle;
                CUdeviceptr            d_gas_output_buffer;
                {
                    cudaEvent_t g_up0, g_up1, g_b0, g_b1, g_c0, g_c1;
                    CUDA_CHECK( cudaEventCreate( &g_up0 ) ); CUDA_CHECK( cudaEventCreate( &g_up1 ) );
                    CUDA_CHECK( cudaEventCreate( &g_b0 ) ); CUDA_CHECK( cudaEventCreate( &g_b1 ) );
                    CUDA_CHECK( cudaEventCreate( &g_c0 ) ); CUDA_CHECK( cudaEventCreate( &g_c1 ) );

                    CUdeviceptr d_aabb_buffer;
                    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb_buffer ), (size_t)n * sizeof( OptixAabb ) ) );
                    launch_fill_aabbs( reinterpret_cast<OptixAabb*>( d_aabb_buffer ), d_p, n, bwf, 0 );
                    CUDA_CHECK( cudaDeviceSynchronize() );
                    CUDA_CHECK( cudaEventRecord( g_up0 ) );
                    launch_fill_aabbs( reinterpret_cast<OptixAabb*>( d_aabb_buffer ), d_p, n, bwf, 0 );
                    CUDA_CHECK( cudaEventRecord( g_up1 ) );

                    OptixAccelBuildOptions accel_options = {};
                    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
                    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

                    uint32_t aabb_input_flags[1] = { OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL };
                    OptixBuildInput aabb_input = {};
                    aabb_input.type                               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
                    aabb_input.customPrimitiveArray.aabbBuffers   = &d_aabb_buffer;
                    aabb_input.customPrimitiveArray.numPrimitives = (unsigned int)n;
                    aabb_input.customPrimitiveArray.flags         = aabb_input_flags;
                    aabb_input.customPrimitiveArray.numSbtRecords = 1;

                    OptixAccelBufferSizes gas_buffer_sizes;
                    CUDA_CHECK( cudaEventRecord( g_b0 ) );
                    OPTIX_CHECK( optixAccelComputeMemoryUsage( context, &accel_options, &aabb_input, 1, &gas_buffer_sizes ) );

                    CUdeviceptr d_temp_buffer_gas;
                    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer_gas ), gas_buffer_sizes.tempSizeInBytes ) );
                    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
                    size_t compactedSizeOffset = ( ( gas_buffer_sizes.outputSizeInBytes + 7ull ) / 8ull ) * 8ull;
                    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ),
                                            compactedSizeOffset + 8 ) );

                    OptixAccelEmitDesc emitProperty = {};
                    emitProperty.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
                    emitProperty.result = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

                    OPTIX_CHECK( optixAccelBuild( context, 0, &accel_options, &aabb_input, 1,
                                                  d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes,
                                                  d_buffer_temp_output_gas_and_compacted_size, gas_buffer_sizes.outputSizeInBytes,
                                                  &gas_handle, &emitProperty, 1 ) );
                    CUDA_CHECK( cudaEventRecord( g_b1 ) );

                    CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );
                    CUDA_CHECK( cudaFree( (void*)d_aabb_buffer ) );

                    size_t compacted_gas_size;
                    CUDA_CHECK( cudaEventRecord( g_c0 ) );
                    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof( size_t ), cudaMemcpyDeviceToHost ) );
                    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
                    {
                        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_gas_output_buffer ), compacted_gas_size ) );
                        OPTIX_CHECK( optixAccelCompact( context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle ) );
                        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
                    }
                    else
                        d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
                    CUDA_CHECK( cudaEventRecord( g_c1 ) );

                    CUDA_CHECK( cudaEventSynchronize( g_c1 ) );
                    float f_up = 0.f, f_b = 0.f, f_c = 0.f;
                    CUDA_CHECK( cudaEventElapsedTime( &f_up, g_up0, g_up1 ) );
                    CUDA_CHECK( cudaEventElapsedTime( &f_b, g_b0, g_b1 ) );
                    CUDA_CHECK( cudaEventElapsedTime( &f_c, g_c0, g_c1 ) );
                    times[g].aabb = f_up; times[g].build = f_b; times[g].compact = f_c;
                    cudaEventDestroy( g_up0 ); cudaEventDestroy( g_up1 );
                    cudaEventDestroy( g_b0 ); cudaEventDestroy( g_b1 );
                    cudaEventDestroy( g_c0 ); cudaEventDestroy( g_c1 );
                }

                // Module / program groups / pipeline / SBT (as in the single-GPU sample).
                OptixModule module = nullptr;
                OptixPipelineCompileOptions pipeline_compile_options = {};
                {
                    OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
                    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
                    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
                    pipeline_compile_options.usesMotionBlur                   = false;
                    pipeline_compile_options.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
                    pipeline_compile_options.numPayloadValues                 = 7;
                    pipeline_compile_options.numAttributeValues               = 3;
                    pipeline_compile_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
                    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
                    pipeline_compile_options.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
                    char   log[2048]; size_t log_size = sizeof( log );
                    OPTIX_CHECK( optixModuleCreate( context, &module_compile_options, &pipeline_compile_options,
                                                    ir, irSize, log, &log_size, &module ) );
                }

                OptixProgramGroup raygen_prog_group, miss_prog_group, hitgroup_prog_group;
                {
                    OptixProgramGroupOptions pgo = {};
                    char log[2048]; size_t log_size;

                    OptixProgramGroupDesc rg = {};
                    rg.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
                    rg.raygen.module            = module;
                    rg.raygen.entryFunctionName = "__raygen__rg";
                    log_size = sizeof( log );
                    OPTIX_CHECK( optixProgramGroupCreate( context, &rg, 1, &pgo, log, &log_size, &raygen_prog_group ) );

                    OptixProgramGroupDesc ms = {};
                    ms.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
                    ms.miss.module            = module;
                    ms.miss.entryFunctionName = "__miss__ms";
                    log_size = sizeof( log );
                    OPTIX_CHECK( optixProgramGroupCreate( context, &ms, 1, &pgo, log, &log_size, &miss_prog_group ) );

                    OptixProgramGroupDesc hg = {};
                    hg.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                    hg.hitgroup.moduleIS            = module;
                    hg.hitgroup.entryFunctionNameIS = "__intersection__sphere";
                    hg.hitgroup.moduleAH            = module;
                    hg.hitgroup.entryFunctionNameAH = "__anyhit__accumulate";
                    log_size = sizeof( log );
                    OPTIX_CHECK( optixProgramGroupCreate( context, &hg, 1, &pgo, log, &log_size, &hitgroup_prog_group ) );
                }

                OptixPipeline pipeline = nullptr;
                {
                    const uint32_t    max_trace_depth  = 1;
                    OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };
                    OptixPipelineLinkOptions link = {};
                    link.maxTraceDepth = max_trace_depth;
                    char log[2048]; size_t log_size = sizeof( log );
                    OPTIX_CHECK( optixPipelineCreate( context, &pipeline_compile_options, &link,
                                                      program_groups, 3, log, &log_size, &pipeline ) );
                    OptixStackSizes stack_sizes = {};
                    for( auto& pg : program_groups )
                        OPTIX_CHECK( optixUtilAccumulateStackSizes( pg, &stack_sizes, pipeline ) );
                    uint32_t s1, s2, s3;
                    OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth, 0, 0, &s1, &s2, &s3 ) );
                    OPTIX_CHECK( optixPipelineSetStackSize( pipeline, s1, s2, s3, 1 ) );
                }

                OptixShaderBindingTable sbt = {};
                {
                    CUdeviceptr rec;
                    RayGenSbtRecord rg_sbt;
                    OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
                    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &rec ), sizeof( RayGenSbtRecord ) ) );
                    CUDA_CHECK( cudaMemcpy( (void*)rec, &rg_sbt, sizeof( rg_sbt ), cudaMemcpyHostToDevice ) );
                    sbt.raygenRecord = rec;

                    MissSbtRecord ms_sbt;
                    OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
                    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &rec ), sizeof( MissSbtRecord ) ) );
                    CUDA_CHECK( cudaMemcpy( (void*)rec, &ms_sbt, sizeof( ms_sbt ), cudaMemcpyHostToDevice ) );
                    sbt.missRecordBase          = rec;
                    sbt.missRecordStrideInBytes = sizeof( MissSbtRecord );
                    sbt.missRecordCount         = 1;

                    HitGroupSbtRecord hg_sbt;
                    OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt ) );
                    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &rec ), sizeof( HitGroupSbtRecord ) ) );
                    CUDA_CHECK( cudaMemcpy( (void*)rec, &hg_sbt, sizeof( hg_sbt ), cudaMemcpyHostToDevice ) );
                    sbt.hitgroupRecordBase          = rec;
                    sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
                    sbt.hitgroupRecordCount         = 1;
                }

                // Shard buffers. out_* are full-size because ray-gen writes by
                // ORIGINAL id; only the shard's range is copied back.
                std::vector<int> h_ids( ns );
                for( int i = 0; i < ns; ++i ) h_ids[i] = s0 + i;

                // C5: Morton-order this shard's seed slots so warp-mates start
                // spatially adjacent (coherent BVH traversal). Complements the
                // contiguous shard split: contiguous keeps each card's working
                // set regional (see --shard note), the in-shard sort adds
                // warp-level coherence on top. Slot -> id via h_ids and the
                // writes go to out[id], so outputs stay byte-identical; the
                // shard's id range is unchanged, so the range copy-back below
                // is untouched. Runs pre-barrier: setup time, not search wall.
                std::vector<float4> h_seed0( hseeds.begin() + s0, hseeds.begin() + s0 + ns );
                const bool do_seedsort = ( args.seedsort < 0 ? ns >= SEEDSORT_AUTO_MIN : args.seedsort != 0 );
                if( do_seedsort && ns > 1 )
                {
                    auto c5_0 = clk::now();
                    const float inv = maxr > 0.0 ? 1.f / (float)maxr : 0.f;
                    std::vector<unsigned long long> key( ns );
                    for( int j = 0; j < ns; ++j )
                        key[j] = morton3d_unit( hseeds[s0 + j].x * inv, hseeds[s0 + j].y * inv,
                                                hseeds[s0 + j].z * inv );
                    std::sort( h_ids.begin(), h_ids.end(),
                               [&]( int a, int b ) { return key[a - s0] < key[b - s0]; } );
                    for( int i = 0; i < ns; ++i ) h_seed0[i] = hseeds[h_ids[i]];
                    times[g].seedsort = ms_since( c5_0, clk::now() );
                }
                int    *d_ids, *d_done;
                float4 *d_cur;
                double *d_outx, *d_outy, *d_outz;
                int    *d_count, *d_conv;
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_ids ), ns * sizeof( int ) ) );
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_done ), ns * sizeof( int ) ) );
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_cur ), ns * sizeof( float4 ) ) );
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_outx ), (size_t)nseeds * sizeof( double ) ) );
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_outy ), (size_t)nseeds * sizeof( double ) ) );
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_outz ), (size_t)nseeds * sizeof( double ) ) );
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_count ), (size_t)nseeds * sizeof( int ) ) );
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_conv ), (size_t)nseeds * sizeof( int ) ) );
                CUDA_CHECK( cudaMemcpy( d_ids, h_ids.data(), ns * sizeof( int ), cudaMemcpyHostToDevice ) );
                CUDA_CHECK( cudaMemcpy( d_cur, h_seed0.data(), ns * sizeof( float4 ), cudaMemcpyHostToDevice ) );

                Params params;
                params.p            = d_p;
                params.num_points   = n;
                params.ids          = d_ids;
                params.cur          = d_cur;
                params.done         = d_done;
                params.n_active     = ns;
                params.chunk        = args.max_iter;
                params.out_x        = d_outx;
                params.out_y        = d_outy;
                params.out_z        = d_outz;
                params.out_count    = d_count;
                params.out_conv     = d_conv;
                params.h2           = (float)( bw * bw );
                params.stop_thresh2 = (float)( stop_thresh * stop_thresh );
                params.handle       = gas_handle;

                CUdeviceptr d_param;
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );

                CUstream stream;
                CUDA_CHECK( cudaStreamCreate( &stream ) );

                // Ready: wait for the global go so the search phase of all GPUs
                // is timed together.
                {
                    std::unique_lock<std::mutex> lk( mtx );
                    ++ready; counted = true;
                    cv_ready.notify_all();
                    cv_go.wait( lk, [&] { return go; } );
                }

                cudaEvent_t t0, t1;
                CUDA_CHECK( cudaEventCreate( &t0 ) );
                CUDA_CHECK( cudaEventCreate( &t1 ) );

                std::vector<int>    h_done( ns );
                std::vector<float4> h_cur( ns );
                int active  = ns;
                int it_done = 0;

                CUDA_CHECK( cudaEventRecord( t0, stream ) );
                while( active > 0 && it_done < args.max_iter )
                {
                    params.n_active = active;
                    params.chunk    = std::min( chunk_cfg, args.max_iter - it_done );
                    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( d_param ), &params, sizeof( params ),
                                                 cudaMemcpyHostToDevice, stream ) );
                    OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( Params ), &sbt, active, 1, 1 ) );
                    it_done += params.chunk;
                    ++times[g].rounds;
                    if( it_done >= args.max_iter )
                        break;

                    CUDA_CHECK( cudaMemcpyAsync( h_done.data(), d_done, active * sizeof( int ),
                                                 cudaMemcpyDeviceToHost, stream ) );
                    CUDA_CHECK( cudaMemcpyAsync( h_cur.data(), d_cur, active * sizeof( float4 ),
                                                 cudaMemcpyDeviceToHost, stream ) );
                    CUDA_CHECK( cudaStreamSynchronize( stream ) );
                    int w = 0;
                    for( int i = 0; i < active; ++i )
                        if( !h_done[i] )
                        {
                            h_ids[w] = h_ids[i];
                            h_cur[w] = h_cur[i];
                            ++w;
                        }
                    if( w == 0 ) { active = 0; break; }
                    if( w != active )
                    {
                        CUDA_CHECK( cudaMemcpyAsync( d_ids, h_ids.data(), w * sizeof( int ),
                                                     cudaMemcpyHostToDevice, stream ) );
                        CUDA_CHECK( cudaMemcpyAsync( d_cur, h_cur.data(), w * sizeof( float4 ),
                                                     cudaMemcpyHostToDevice, stream ) );
                    }
                    active = w;
                }
                CUDA_CHECK( cudaEventRecord( t1, stream ) );
                CUDA_CHECK( cudaStreamSynchronize( stream ) );
                CUDA_CHECK( cudaEventElapsedTime( &times[g].search, t0, t1 ) );

                // Copy the shard's outputs (original-id-indexed range) back.
                CUDA_CHECK( cudaMemcpy( ocx.data() + s0, d_outx + s0, ns * sizeof( double ), cudaMemcpyDeviceToHost ) );
                CUDA_CHECK( cudaMemcpy( ocy.data() + s0, d_outy + s0, ns * sizeof( double ), cudaMemcpyDeviceToHost ) );
                CUDA_CHECK( cudaMemcpy( ocz.data() + s0, d_outz + s0, ns * sizeof( double ), cudaMemcpyDeviceToHost ) );
                CUDA_CHECK( cudaMemcpy( ocount.data() + s0, d_count + s0, ns * sizeof( int ), cudaMemcpyDeviceToHost ) );
                CUDA_CHECK( cudaMemcpy( oconv.data() + s0, d_conv + s0, ns * sizeof( int ), cudaMemcpyDeviceToHost ) );

                cudaEventDestroy( t0 ); cudaEventDestroy( t1 );
                cudaStreamDestroy( stream );
                cudaFree( reinterpret_cast<void*>( d_param ) );
                cudaFree( d_ids ); cudaFree( d_done ); cudaFree( d_cur );
                cudaFree( d_outx ); cudaFree( d_outy ); cudaFree( d_outz ); cudaFree( d_count ); cudaFree( d_conv );
                cudaFree( reinterpret_cast<void*>( sbt.raygenRecord ) );
                cudaFree( reinterpret_cast<void*>( sbt.missRecordBase ) );
                cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) );
                cudaFree( reinterpret_cast<void*>( d_gas_output_buffer ) );
                cudaFree( d_p );
                optixPipelineDestroy( pipeline );
                optixProgramGroupDestroy( hitgroup_prog_group );
                optixProgramGroupDestroy( miss_prog_group );
                optixProgramGroupDestroy( raygen_prog_group );
                optixModuleDestroy( module );
                optixDeviceContextDestroy( context );
            }
            catch( std::exception& e )
            {
                errs[g] = e.what();
                if( !counted )
                {
                    std::lock_guard<std::mutex> lk( mtx );
                    ++ready;
                    cv_ready.notify_all();
                }
            }
        };

        std::vector<std::thread> threads;
        for( int g = 0; g < G; ++g ) threads.emplace_back( worker, g );

        // Wait until every GPU finished its setup, then release them together
        // and wall-clock the whole parallel search phase.
        {
            std::unique_lock<std::mutex> lk( mtx );
            cv_ready.wait( lk, [&] { return ready == G; } );
        }
        auto w0 = clk::now();
        {
            std::lock_guard<std::mutex> lk( mtx );
            go = true;
        }
        cv_go.notify_all();
        for( auto& t : threads ) t.join();
        const double ms_shift = ms_since( w0, clk::now() );

        for( int g = 0; g < G; ++g )
            if( !errs[g].empty() )
            { std::fprintf( stderr, "GPU %d failed: %s\n", g, errs[g].c_str() ); return 1; }

        if( !sperm.empty() )   // undo the strided permutation (back to original ids)
        {
            std::vector<double> tdx( nseeds ), tdy( nseeds ), tdz( nseeds );
            std::vector<int>    tci( nseeds ), tcv( nseeds );
            for( int j = 0; j < nseeds; ++j )
            {
                tdx[sperm[j]] = ocx[j];  tdy[sperm[j]] = ocy[j]; tdz[sperm[j]] = ocz[j];
                tci[sperm[j]] = ocount[j]; tcv[sperm[j]] = oconv[j];
            }
            ocx.swap( tdx ); ocy.swap( tdy ); ocz.swap( tdz ); ocount.swap( tci ); oconv.swap( tcv );
        }

        // Back to original coordinate space.
        for( int i = 0; i < nseeds; ++i ) { ocx[i] += offx; ocy[i] += offy; ocz[i] += offz; }

        int n_unconverged = 0;
        for( int i = 0; i < nseeds; ++i ) n_unconverged += ( oconv[i] == 0 );
        std::printf( "Seeds: %d total, %d did NOT converge within max_iter=%d.\n", nseeds, n_unconverged, args.max_iter );

        std::vector<double> ux, uy, uz;
        auto md0 = clk::now();
        dedup_centers( ocx, ocy, ocz, ocount, bw, ux, uy, uz );
        t_dedup = ms_since( md0, clk::now() );
        const int k = (int)ux.size();

        std::vector<int> labels;
        assign_labels( xs, ys, zs, ux, uy, uz, labels );

        std::vector<int> counts( k, 0 );
        for( int i = 0; i < n; ++i ) counts[labels[i]]++;
        std::vector<int> order( k );
        for( int i = 0; i < k; ++i ) order[i] = i;
        std::stable_sort( order.begin(), order.end(), [&]( int a, int b ) { return counts[a] > counts[b]; } );

        double t_build_total = 0.0;
        std::printf( "\nPer-GPU setup + search (concurrent):\n" );
        for( int g = 0; g < G; ++g )
        {
            const double b = times[g].upload + times[g].reorder + times[g].aabb + times[g].build + times[g].compact + times[g].seedsort;
            t_build_total = std::max( t_build_total, b );
            std::printf( "  GPU %d: %7d seeds | upload %6.2f + morton %6.2f + aabb %5.2f + build %6.2f + compact %5.2f + seedsort %6.2f = bvh %7.2f ms | search %8.2f ms (%d rounds)\n",
                         g, times[g].nseeds, times[g].upload, times[g].reorder, times[g].aabb, times[g].build, times[g].compact, times[g].seedsort, b,
                         times[g].search, times[g].rounds );
        }
        // Phase sum convention: bvh = slowest GPU's setup (they overlap), search
        // = wall clock of the barrier-released parallel launch phase.
        const double t_meanshift = t_build_total + ms_shift;
        const double t_sum = t_estimate + t_binning + t_meanshift + t_dedup;
        std::printf( "Mean-shift (OptiX, %d GPUs) search wall: %.2f ms.\n", G, ms_shift );
        std::printf( "PHASE ms: estimate=%.2f binning=%.2f meanshift=%.2f "
                     "(bvh=%.2f + search=%.2f) modemerge=%.2f | SUM=%.2f\n",
                     t_estimate, t_binning, t_meanshift, t_build_total,
                     ms_shift, t_dedup, t_sum );
        std::printf( "Found %d clusters.\n\n", k );
        std::printf( "%7s  %8s  %14s  %14s\n", "cluster", "count", "center_x", "center_y" );
        for( int oi = 0; oi < k && oi < 50; ++oi )
        {
            const int c = order[oi];
            std::printf( "%7d  %8d  %14.1f  %14.1f\n", c, counts[c], ux[c], uy[c] );
        }
        if( k > 50 ) std::printf( "... (%d more clusters)\n", k - 50 );

        std::ofstream out( args.output );
        out << "CASE#,X,Y,Z,cluster_label\n";
        for( int i = 0; i < n; ++i )
            out << cases[i] << ',' << (long long)std::llround( xs[i] ) << ','
                << (long long)std::llround( ys[i] ) << ',' << (long long)std::llround( zs[i] )
                << ',' << labels[i] << '\n';
        out.close();
        std::printf( "\nWrote per-point labels to %s\n", args.output.c_str() );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
