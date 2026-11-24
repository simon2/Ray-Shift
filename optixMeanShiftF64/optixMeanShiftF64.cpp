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

// Ray-traced mean-shift clustering of Chicago crime X/Y coordinates (feet).
//
// Same pipeline and I/O contract as ../cuda-ms, but the mean-shift stage is a
// ray-tracing kernel instead of a hand-written CUDA kernel:
//   1. estimate_bandwidth (host, optional)
//   2. bin seeding (host)
//   3. mean-shift via OptiX  -- one ray-gen thread per seed, disks = neighbour index
//   4. dedup (host)          -- merge centers within bandwidth
//   5. assign (host)         -- label = nearest surviving center
//
// Build with the rest of the samples (see CMakeLists.txt). Run:
//   ./optixMeanShift --sample 10000 --bandwidth 1500

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include "optixMeanShiftF64.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

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

// Defined in aabb_kernel.cu (compiled as a regular CUDA object): generates one
// disk AABB per data point on the GPU from the device point coordinates.
extern "C" void launch_fill_aabbs( OptixAabb* d_aabb, const float2* d_p,
                                   int n, float half_extent, cudaStream_t stream );
extern "C" void launch_morton_reorder( float2* d_p, int n, float maxr, cudaStream_t stream );


// ---------------------------------------------------------------------------
// Host helpers (ported from ../cuda-ms/meanshift.cu; identical algorithm)
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
    int morton = 1;            // B6: Morton-reorder the device point array (byte-exact)
    int seedsort = -1;         // C5: Morton-order the seed slots (byte-exact). -1=auto, 0=off, 1=on
};

// C5: below this many seeds the launch underfills the GPU, and sorting the
// slots concentrates the dense seeds into a few warps (load imbalance) instead
// of spreading them -- a net loss. Above it, warp coherence wins big. The
// crossover was measured at ~32k seeds on an RTX A6000 (84 SMs); auto mode
// sorts only past it, so bin-seeding (few seeds) never regresses.
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
        else if( k == "--morton" ) a.morton = std::stoi( next( "--morton" ) );
        else if( k == "--seedsort" ) a.seedsort = std::stoi( next( "--seedsort" ) );
        else if( k == "-h" || k == "--help" )
        {
            std::printf( "Usage: %s [--input f] [--output f] [--sample N|all] "
                         "[--bandwidth ft] [--quantile q] [--seed s] [--all-seeds] "
                         "[--max-iter N]\n", argv[0] );
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
                     std::vector<double>& xs, std::vector<double>& ys )
{
    std::ifstream in( path );
    if( !in ) { std::fprintf( stderr, "Could not open input: %s\n", path.c_str() ); std::exit( 1 ); }
    std::string line;
    if( !std::getline( in, line ) ) { std::fprintf( stderr, "Empty input file\n" ); std::exit( 1 ); }

    int ci_case = -1, ci_x = -1, ci_y = -1, idx = 0;
    {
        std::stringstream hs( line );
        std::string col;
        while( std::getline( hs, col, ',' ) )
        {
            if( col == "CASE#" ) ci_case = idx;
            else if( col == "X COORDINATE" ) ci_x = idx;
            else if( col == "Y COORDINATE" ) ci_y = idx;
            ++idx;
        }
    }
    if( ci_case < 0 || ci_x < 0 || ci_y < 0 )
    { std::fprintf( stderr, "Header missing CASE#/X COORDINATE/Y COORDINATE columns\n" ); std::exit( 1 ); }
    const int need = std::max( ci_case, std::max( ci_x, ci_y ) );

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
        double x, y;
        if( !parse_double( f[ci_x], x ) || !parse_double( f[ci_y], y ) ) continue;
        cases.push_back( f[ci_case] );
        xs.push_back( x );
        ys.push_back( y );
    }
    std::printf( "Loaded %ld rows, %zu valid after dropping missing coordinates.\n", before, xs.size() );
}

static double estimate_bandwidth( const std::vector<double>& xs, const std::vector<double>& ys,
                                  double quantile, unsigned seed )
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
        const double x = xs[idx[a]], y = ys[idx[a]];
        for( int b = 0; b < n_sub; ++b )
        {
            const double dx = xs[idx[b]] - x, dy = ys[idx[b]] - y;
            d2[b] = dx * dx + dy * dy;
        }
        std::nth_element( d2.begin(), d2.begin() + ( k - 1 ), d2.end() );
        acc += std::sqrt( d2[k - 1] );
    }
    return acc / (double)n_sub;
}

static void bin_seeds( const std::vector<double>& xs, const std::vector<double>& ys, double bw,
                       std::vector<double>& sx, std::vector<double>& sy )
{
    std::unordered_map<long long, int> bins;
    bins.reserve( xs.size() );
    for( size_t i = 0; i < xs.size(); ++i )
    {
        const long long bx = (long long)std::llround( xs[i] / bw );
        const long long by = (long long)std::llround( ys[i] / bw );
        const long long key = ( bx << 32 ) ^ ( by & 0xffffffffLL );
        bins[key]++;
    }
    for( const auto& kv : bins )
    {
        const long long key = kv.first;
        const int bx = (int)( key >> 32 );
        const int by = (int)( key & 0xffffffffLL );
        sx.push_back( bx * bw );
        sy.push_back( by * bw );
    }
}

static void dedup_centers( const std::vector<double>& cx, const std::vector<double>& cy,
                           const std::vector<int>& dens, double bw,
                           std::vector<double>& ux, std::vector<double>& uy )
{
    const int m = (int)cx.size();
    std::vector<int> order( m );
    for( int i = 0; i < m; ++i ) order[i] = i;
    // Tie-break mirrors sklearn's sorted(..., key=lambda tup: (tup[1], tup[0]),
    // reverse=True): density descending, then the center tuple descending. With
    // density alone the surviving set depends on the seed array order, so an
    // engine that permutes its seeds (cluster's strided --dist discrete) derives
    // a different partition from the very same modes.
    std::stable_sort( order.begin(), order.end(), [&]( int a, int b ) {
        if( dens[a] != dens[b] ) return dens[a] > dens[b];
        if( cx[a] != cx[b] )     return cx[a] > cx[b];
        return cy[a] > cy[b];
    } );
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
            const double dx = cx[i] - cx[j], dy = cy[i] - cy[j];
            if( dx * dx + dy * dy <= bw2 ) unique[j] = 0;
        }
        unique[i] = 1;
    }
    for( int i = 0; i < m; ++i )
        if( unique[i] ) { ux.push_back( cx[i] ); uy.push_back( cy[i] ); }
}

// One label per point: nearest surviving center (host; O(N*K), K small).
static void assign_labels( const std::vector<double>& xs, const std::vector<double>& ys,
                           const std::vector<double>& ux, const std::vector<double>& uy,
                           std::vector<int>& labels )
{
    const int n = (int)xs.size(), k = (int)ux.size();
    labels.assign( n, 0 );
    for( int i = 0; i < n; ++i )
    {
        double best = 1e308; int bl = 0;
        for( int c = 0; c < k; ++c )
        {
            const double dx = xs[i] - ux[c], dy = ys[i] - uy[c];
            const double d = dx * dx + dy * dy;
            if( d < best ) { best = d; bl = c; }
        }
        labels[i] = bl;
    }
}


// C5: 2D Morton code of a point already normalized to [0,1]^2, on a 21-bit grid
// per axis. Only used to order the seed slots, so the grid resolution is a
// locality knob and has no effect on results.
static unsigned long long morton2d_unit( float sx, float sy )
{
    auto part1by1 = []( unsigned int x ) -> unsigned long long {
        unsigned long long v = x;
        v = ( v | ( v << 16 ) ) & 0x0000FFFF0000FFFFull;
        v = ( v | ( v << 8 ) )  & 0x00FF00FF00FF00FFull;
        v = ( v | ( v << 4 ) )  & 0x0F0F0F0F0F0F0F0Full;
        v = ( v | ( v << 2 ) )  & 0x3333333333333333ull;
        v = ( v | ( v << 1 ) )  & 0x5555555555555555ull;
        return v;
    };
    sx = std::min( std::max( sx, 0.f ), 1.f );
    sy = std::min( std::max( sy, 0.f ), 1.f );
    return ( part1by1( (unsigned int)( sx * 2097151.f ) ) << 1 ) | part1by1( (unsigned int)( sy * 2097151.f ) );
}


static void context_log_cb( unsigned int level, const char* tag, const char* message, void* )
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}


// ---------------------------------------------------------------------------

int main( int argc, char** argv )
{
    Args args = parse_args( argc, argv );

    std::vector<std::string> cases;
    std::vector<double> xs, ys;
    load_xy( args.input, cases, xs, ys );

    // Sampling: deterministic seeded shuffle (mirrors cuda-ms).
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
    std::printf( "Clustering %d points on X/Y coordinates (feet).\n", n );
    if( n == 0 ) { std::fprintf( stderr, "No points to cluster.\n" ); return 1; }

    // Per-phase host timers (ms). Summed at the end (see phase report).
    using clk = std::chrono::steady_clock;
    auto ms_since = []( clk::time_point a, clk::time_point b ) {
        return std::chrono::duration<double, std::milli>( b - a ).count();
    };
    double t_estimate = 0.0, t_binning = 0.0, t_dedup = 0.0;

    double bw = args.bandwidth;
    if( bw <= 0.0 )
    {
        auto e0 = clk::now();
        bw = estimate_bandwidth( xs, ys, args.quantile, args.seed );
        t_estimate = ms_since( e0, clk::now() );
        std::printf( "Estimated bandwidth: %.2f feet (quantile=%.2f).\n", bw, args.quantile );
    }
    else std::printf( "Using bandwidth: %.2f feet.\n", bw );
    if( bw <= 0.0 ) { std::fprintf( stderr, "Bandwidth is non-positive.\n" ); return 1; }

    // ---- seeds ----
    std::vector<double> sx, sy;
    if( args.all_seeds )
    {
        sx = xs; sy = ys;
        std::printf( "All-seeds: %d seeds (one per point), no bin seeding.\n", n );
    }
    else
    {
        auto b0 = clk::now();
        bin_seeds( xs, ys, bw, sx, sy );
        t_binning = ms_since( b0, clk::now() );
        std::printf( "Bin seeding: %zu seeds from %d points.\n", sx.size(), n );
    }
    const int nseeds = (int)sx.size();

    // ---- recenter (subtract integer min) so recentered coords stay exact in float ----
    const double offx = *std::min_element( xs.begin(), xs.end() );
    const double offy = *std::min_element( ys.begin(), ys.end() );
    const double maxr = std::max( *std::max_element( xs.begin(), xs.end() ) - offx,
                                  *std::max_element( ys.begin(), ys.end() ) - offy );

    std::vector<float> hpx( n ), hpy( n );
    for( int i = 0; i < n; ++i ) { hpx[i] = (float)( xs[i] - offx ); hpy[i] = (float)( ys[i] - offy ); }
    std::vector<float> hcurx( nseeds ), hcury( nseeds );
    for( int i = 0; i < nseeds; ++i ) { hcurx[i] = (float)( sx[i] - offx ); hcury[i] = (float)( sy[i] - offy ); }

    const double stop_thresh = 1e-3 * bw;

    try
    {
        //
        // Initialize CUDA and create OptiX context
        //
        OptixDeviceContext context = nullptr;
        {
            CUDA_CHECK( cudaFree( 0 ) );
            CUcontext cuCtx = 0;
            OPTIX_CHECK( optixInit() );
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction = &context_log_cb;
            options.logCallbackLevel    = 4;
            OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );
        }

        // Geometry + BVH build timing (filled below).
        double t_pts_upload = 0.0, t_aabb_gen = 0.0, t_accel_build = 0.0, t_accel_compact = 0.0;

        // Upload the recentered points up front (packed float2, B2): needed both
        // to generate the disk AABBs on the GPU and for the mean-shift launch.
        std::vector<float2> hp( n );
        for( int i = 0; i < n; ++i ) hp[i] = make_float2( hpx[i], hpy[i] );
        float2* d_p = nullptr;
        {
            cudaEvent_t p0, p1;
            CUDA_CHECK( cudaEventCreate( &p0 ) ); CUDA_CHECK( cudaEventCreate( &p1 ) );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_p ), n * sizeof( float2 ) ) );
            CUDA_CHECK( cudaEventRecord( p0 ) );
            CUDA_CHECK( cudaMemcpy( d_p, hp.data(), n * sizeof( float2 ), cudaMemcpyHostToDevice ) );
            CUDA_CHECK( cudaEventRecord( p1 ) );
            CUDA_CHECK( cudaEventSynchronize( p1 ) );
            float f = 0.f; CUDA_CHECK( cudaEventElapsedTime( &f, p0, p1 ) ); t_pts_upload = f;
            CUDA_CHECK( cudaEventDestroy( p0 ) ); CUDA_CHECK( cudaEventDestroy( p1 ) );
        }

        // B6: Morton-reorder the device copy (host arrays keep the original
        // order; results are order-invariant, so this is byte-exact).
        double t_reorder = 0.0;
        if( args.morton )
        {
            cudaEvent_t r0, r1;
            CUDA_CHECK( cudaEventCreate( &r0 ) ); CUDA_CHECK( cudaEventCreate( &r1 ) );
            CUDA_CHECK( cudaEventRecord( r0 ) );
            launch_morton_reorder( d_p, n, (float)maxr, 0 );
            CUDA_CHECK( cudaEventRecord( r1 ) );
            CUDA_CHECK( cudaEventSynchronize( r1 ) );
            float f = 0.f; CUDA_CHECK( cudaEventElapsedTime( &f, r0, r1 ) ); t_reorder = f;
            CUDA_CHECK( cudaEventDestroy( r0 ) ); CUDA_CHECK( cudaEventDestroy( r1 ) );
        }

        //
        // Build the GAS: one flat disk (custom AABB) per data point at z=0.
        // The AABB is exactly the disk's bounding square -- no pad: with
        // integer-valued recentered coordinates and an integer bandwidth the
        // box endpoints (x +/- bw) are exact in float, so the hardware cull's
        // per-axis comparisons are exact; the intersection program makes the
        // in-window decision.
        //
        OptixTraversableHandle gas_handle;
        CUdeviceptr            d_gas_output_buffer;
        {
            const float bwf = (float)bw;

            cudaEvent_t g_up0, g_up1, g_b0, g_b1, g_c0, g_c1;
            CUDA_CHECK( cudaEventCreate( &g_up0 ) ); CUDA_CHECK( cudaEventCreate( &g_up1 ) );
            CUDA_CHECK( cudaEventCreate( &g_b0  ) ); CUDA_CHECK( cudaEventCreate( &g_b1  ) );
            CUDA_CHECK( cudaEventCreate( &g_c0  ) ); CUDA_CHECK( cudaEventCreate( &g_c1  ) );

            // Geometry construction on the GPU: one disk AABB per point, built
            // directly from the device coordinates (no host fill, no AABB upload).
            CUdeviceptr d_aabb_buffer;
            const size_t aabb_bytes = (size_t)n * sizeof( OptixAabb );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb_buffer ), aabb_bytes ) );
            // Warm-up: the first launch of this kernel pays a one-time lazy
            // module-load cost; warming up makes the timed launch reflect the
            // steady-state per-build cost.
            launch_fill_aabbs( reinterpret_cast<OptixAabb*>( d_aabb_buffer ), d_p, n, bwf + 1.0f, 0 );  // DC: +1 integer pad absorbs fl32(double center) offset
            CUDA_CHECK( cudaDeviceSynchronize() );
            CUDA_CHECK( cudaEventRecord( g_up0 ) );
            launch_fill_aabbs( reinterpret_cast<OptixAabb*>( d_aabb_buffer ), d_p, n, bwf + 1.0f, 0 );  // DC: +1 integer pad absorbs fl32(double center) offset
            CUDA_CHECK( cudaEventRecord( g_up1 ) );

            OptixAccelBuildOptions accel_options = {};
            // PREFER_FAST_TRACE was A/B-tested here and made no measurable
            // difference for custom-primitive AABBs (see OPTIMIZATIONS.md B4).
            accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
            accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

            // SINGLE_ANYHIT is mandatory for exact counting: without it OptiX
            // may invoke the any-hit program more than once per primitive.
            uint32_t aabb_input_flags[1] = { OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL };
            OptixBuildInput aabb_input = {};
            aabb_input.type                               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
            aabb_input.customPrimitiveArray.aabbBuffers   = &d_aabb_buffer;
            aabb_input.customPrimitiveArray.numPrimitives = (unsigned int)n;
            aabb_input.customPrimitiveArray.flags         = aabb_input_flags;
            aabb_input.customPrimitiveArray.numSbtRecords = 1;

            // BVH build (accel structure).
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

            // BVH compaction.
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
            {
                d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
            }
            CUDA_CHECK( cudaEventRecord( g_c1 ) );

            // All accel work is async on the default stream; sync before reading times.
            CUDA_CHECK( cudaEventSynchronize( g_c1 ) );
            float f_up = 0.f, f_b = 0.f, f_c = 0.f;
            CUDA_CHECK( cudaEventElapsedTime( &f_up, g_up0, g_up1 ) );
            CUDA_CHECK( cudaEventElapsedTime( &f_b,  g_b0,  g_b1  ) );
            CUDA_CHECK( cudaEventElapsedTime( &f_c,  g_c0,  g_c1  ) );
            t_aabb_gen = f_up; t_accel_build = f_b; t_accel_compact = f_c;
            cudaEventDestroy( g_up0 ); cudaEventDestroy( g_up1 );
            cudaEventDestroy( g_b0  ); cudaEventDestroy( g_b1  );
            cudaEventDestroy( g_c0  ); cudaEventDestroy( g_c1  );
        }

        //
        // Create module
        //
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
            pipeline_compile_options.numPayloadValues                 = 9;   // DC: cx,cy double(2 each) + sx,sy int64(2 each) + n
            pipeline_compile_options.numAttributeValues               = 2;   // B2: point coords IS -> AH
            pipeline_compile_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
            pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
            pipeline_compile_options.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

            size_t      inputSize = 0;
            const char* input = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixMeanShiftF64.cu", inputSize );
            OPTIX_CHECK_LOG( optixModuleCreate( context, &module_compile_options, &pipeline_compile_options,
                                                input, inputSize, LOG, &LOG_SIZE, &module ) );
        }

        //
        // Create program groups: raygen, miss, hitgroup (IS + AH, no CH)
        //
        OptixProgramGroup raygen_prog_group   = nullptr;
        OptixProgramGroup miss_prog_group     = nullptr;
        OptixProgramGroup hitgroup_prog_group = nullptr;
        {
            OptixProgramGroupOptions program_group_options = {};

            OptixProgramGroupDesc raygen_prog_group_desc = {};
            raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module            = module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
            OPTIX_CHECK_LOG( optixProgramGroupCreate( context, &raygen_prog_group_desc, 1,
                                                      &program_group_options, LOG, &LOG_SIZE, &raygen_prog_group ) );

            OptixProgramGroupDesc miss_prog_group_desc = {};
            miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_prog_group_desc.miss.module            = module;
            miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
            OPTIX_CHECK_LOG( optixProgramGroupCreate( context, &miss_prog_group_desc, 1,
                                                      &program_group_options, LOG, &LOG_SIZE, &miss_prog_group ) );

            OptixProgramGroupDesc hitgroup_prog_group_desc = {};
            hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hitgroup_prog_group_desc.hitgroup.moduleIS            = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__disk";
            hitgroup_prog_group_desc.hitgroup.moduleAH            = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__accumulate";
            hitgroup_prog_group_desc.hitgroup.moduleCH            = nullptr;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
            OPTIX_CHECK_LOG( optixProgramGroupCreate( context, &hitgroup_prog_group_desc, 1,
                                                      &program_group_options, LOG, &LOG_SIZE, &hitgroup_prog_group ) );
        }

        //
        // Link pipeline
        //
        OptixPipeline pipeline = nullptr;
        {
            const uint32_t    max_trace_depth  = 1;
            OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth = max_trace_depth;
            OPTIX_CHECK_LOG( optixPipelineCreate( context, &pipeline_compile_options, &pipeline_link_options,
                                                  program_groups, sizeof( program_groups ) / sizeof( program_groups[0] ),
                                                  LOG, &LOG_SIZE, &pipeline ) );

            OptixStackSizes stack_sizes = {};
            for( auto& prog_group : program_groups )
                OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes, pipeline ) );

            uint32_t dc_stack_from_trav, dc_stack_from_state, cont_stack;
            OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth, 0, 0,
                                                     &dc_stack_from_trav, &dc_stack_from_state, &cont_stack ) );
            OPTIX_CHECK( optixPipelineSetStackSize( pipeline, dc_stack_from_trav, dc_stack_from_state, cont_stack, 1 ) );
        }

        //
        // Shader binding table
        //
        OptixShaderBindingTable sbt = {};
        {
            CUdeviceptr  raygen_record;
            const size_t raygen_record_size = sizeof( RayGenSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
            RayGenSbtRecord rg_sbt;
            OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
            CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( raygen_record ), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice ) );

            CUdeviceptr  miss_record;
            const size_t miss_record_size = sizeof( MissSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
            MissSbtRecord ms_sbt;
            OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
            CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( miss_record ), &ms_sbt, miss_record_size, cudaMemcpyHostToDevice ) );

            CUdeviceptr  hitgroup_record;
            const size_t hitgroup_record_size = sizeof( HitGroupSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), hitgroup_record_size ) );
            HitGroupSbtRecord hg_sbt;
            OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt ) );
            CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( hitgroup_record ), &hg_sbt, hitgroup_record_size, cudaMemcpyHostToDevice ) );

            sbt.raygenRecord                = raygen_record;
            sbt.missRecordBase              = miss_record;
            sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
            sbt.missRecordCount             = 1;
            sbt.hitgroupRecordBase          = hitgroup_record;
            sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
            sbt.hitgroupRecordCount         = 1;
        }

        //
        // Device buffers for the mean-shift launch
        //
        // d_p was allocated and uploaded earlier (for AABB generation).
        // C2 buffers: slot -> original id map, moving centers, done flags.
        // C5: order the seed slots along a Morton curve. A warp shares one RT
        // pipeline; when consecutive launch indices are consecutive rows of the
        // input file, its 32 rays start at 32 unrelated places and descend 32
        // disjoint BVH branches. Sorting the slots spatially makes warp-mates
        // traverse near-identical node sets and touch the same p[] cache lines.
        // Slot -> original id goes through ids[], and every result is written to
        // out_*[orig], so the outputs are byte-identical to the unsorted launch.
        std::vector<int> h_ids( nseeds );
        for( int i = 0; i < nseeds; ++i ) h_ids[i] = i;
        const bool do_seedsort = ( args.seedsort < 0 ? nseeds >= SEEDSORT_AUTO_MIN : args.seedsort != 0 );
        double t_seedsort = 0.0;
        if( do_seedsort && nseeds > 1 )
        {
            auto s0 = clk::now();
            const float inv = maxr > 0.0 ? 1.f / (float)maxr : 0.f;
            std::vector<unsigned long long> key( nseeds );
            for( int i = 0; i < nseeds; ++i ) key[i] = morton2d_unit( hcurx[i] * inv, hcury[i] * inv );
            std::sort( h_ids.begin(), h_ids.end(), [&]( int a, int b ) { return key[a] < key[b]; } );
            t_seedsort = ms_since( s0, clk::now() );
        }
        std::vector<double2> hseeds( nseeds );
        for( int i = 0; i < nseeds; ++i )
            hseeds[i] = make_double2( (double)hcurx[h_ids[i]], (double)hcury[h_ids[i]] );
        int    *d_ids, *d_done;
        double2 *d_cur;
        double *d_outx, *d_outy;
        int    *d_count, *d_conv;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_ids ), nseeds * sizeof( int ) ) );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_done ), nseeds * sizeof( int ) ) );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_cur ), nseeds * sizeof( double2 ) ) );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_outx ), nseeds * sizeof( double ) ) );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_outy ), nseeds * sizeof( double ) ) );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_count ), nseeds * sizeof( int ) ) );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_conv ), nseeds * sizeof( int ) ) );
        CUDA_CHECK( cudaMemcpy( d_ids, h_ids.data(), nseeds * sizeof( int ), cudaMemcpyHostToDevice ) );
        CUDA_CHECK( cudaMemcpy( d_cur, hseeds.data(), nseeds * sizeof( double2 ), cudaMemcpyHostToDevice ) );

        Params params;
        params.p            = d_p;
        params.num_points   = n;
        params.ids          = d_ids;
        params.cur          = d_cur;
        params.done         = d_done;
        params.n_active     = nseeds;
        params.chunk        = args.max_iter;
        params.out_x        = d_outx;
        params.out_y        = d_outy;
        params.out_count    = d_count;
        params.out_conv     = d_conv;
        params.h2           = bw * bw;
        params.stop_thresh2 = stop_thresh * stop_thresh;
        params.handle       = gas_handle;

        CUdeviceptr d_param;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );

        //
        // Launch (C2): host-driven rounds of `chunk` iterations, compacting
        // converged seeds away between rounds so live trajectories stay packed
        // into dense warps. chunk = max_iter (the default) is a single launch,
        // identical to the pre-C2 behaviour.
        //
        CUstream stream;
        CUDA_CHECK( cudaStreamCreate( &stream ) );

        cudaEvent_t t0, t1;
        CUDA_CHECK( cudaEventCreate( &t0 ) );
        CUDA_CHECK( cudaEventCreate( &t1 ) );

        const int chunk_cfg = args.chunk > 0 ? args.chunk : args.max_iter;
        std::vector<int>    h_done( nseeds );
        std::vector<double2> h_cur( nseeds );
        int active  = nseeds;
        int it_done = 0;
        int rounds  = 0;

        CUDA_CHECK( cudaEventRecord( t0, stream ) );
        while( active > 0 && it_done < args.max_iter )
        {
            params.n_active = active;
            params.chunk    = std::min( chunk_cfg, args.max_iter - it_done );
            CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( d_param ), &params, sizeof( params ),
                                         cudaMemcpyHostToDevice, stream ) );
            OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( Params ), &sbt, active, 1, 1 ) );
            it_done += params.chunk;
            ++rounds;
            if( it_done >= args.max_iter )
                break;

            // Compact: drop converged slots, keep (id, center) of the rest.
            CUDA_CHECK( cudaMemcpyAsync( h_done.data(), d_done, active * sizeof( int ),
                                         cudaMemcpyDeviceToHost, stream ) );
            CUDA_CHECK( cudaMemcpyAsync( h_cur.data(), d_cur, active * sizeof( double2 ),
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
            if( w == 0 )
            {
                active = 0;
                break;
            }
            if( w != active )
            {
                CUDA_CHECK( cudaMemcpyAsync( d_ids, h_ids.data(), w * sizeof( int ),
                                             cudaMemcpyHostToDevice, stream ) );
                CUDA_CHECK( cudaMemcpyAsync( d_cur, h_cur.data(), w * sizeof( double2 ),
                                             cudaMemcpyHostToDevice, stream ) );
            }
            active = w;
        }
        CUDA_CHECK( cudaEventRecord( t1, stream ) );
        CUDA_SYNC_CHECK();

        float ms_shift = 0.f;
        CUDA_CHECK( cudaEventElapsedTime( &ms_shift, t0, t1 ) );
        if( chunk_cfg < args.max_iter )
            std::printf( "C2: chunk=%d, %d rounds.\n", chunk_cfg, rounds );

        std::vector<double> ocx( nseeds ), ocy( nseeds );
        std::vector<int>    ocount( nseeds ), oconv( nseeds );
        CUDA_CHECK( cudaMemcpy( ocx.data(), d_outx, nseeds * sizeof( double ), cudaMemcpyDeviceToHost ) );
        CUDA_CHECK( cudaMemcpy( ocy.data(), d_outy, nseeds * sizeof( double ), cudaMemcpyDeviceToHost ) );
        CUDA_CHECK( cudaMemcpy( ocount.data(), d_count, nseeds * sizeof( int ), cudaMemcpyDeviceToHost ) );
        CUDA_CHECK( cudaMemcpy( oconv.data(), d_conv, nseeds * sizeof( int ), cudaMemcpyDeviceToHost ) );

        // Back to original coordinate space.
        for( int i = 0; i < nseeds; ++i ) { ocx[i] += offx; ocy[i] += offy; }

        int n_unconverged = 0;
        for( int i = 0; i < nseeds; ++i ) n_unconverged += ( oconv[i] == 0 );
        std::printf( "Seeds: %d total, %d did NOT converge within max_iter=%d.\n", nseeds, n_unconverged, args.max_iter );

        // ---- dedup + assign (host) ----  [dedup = mode merge]
        std::vector<double> ux, uy;
        auto md0 = clk::now();
        dedup_centers( ocx, ocy, ocount, bw, ux, uy );
        t_dedup = ms_since( md0, clk::now() );
        const int k = (int)ux.size();

        std::vector<int> labels;
        assign_labels( xs, ys, ux, uy, labels );

        // ---- report ----
        std::vector<int> counts( k, 0 );
        for( int i = 0; i < n; ++i ) counts[labels[i]]++;
        std::vector<int> order( k );
        for( int i = 0; i < k; ++i ) order[i] = i;
        std::stable_sort( order.begin(), order.end(), [&]( int a, int b ) { return counts[a] > counts[b]; } );

        const double t_build_total = t_pts_upload + t_reorder + t_aabb_gen + t_accel_build + t_accel_compact + t_seedsort;
        std::printf( "\nGeometry + BVH build breakdown:\n" );
        std::printf( "  points upload (H2D): %8.2f ms  (also reused by the search)\n", t_pts_upload );
        std::printf( "  Morton reorder     : %8.2f ms  (--morton %d)\n", t_reorder, args.morton );
        std::printf( "  seed Morton sort   : %8.2f ms  (--seedsort %d -> %s)\n",
                     t_seedsort, args.seedsort, do_seedsort ? "on" : "off" );
        std::printf( "  AABB gen (GPU)     : %8.2f ms\n", t_aabb_gen );
        std::printf( "  accel build        : %8.2f ms\n", t_accel_build );
        std::printf( "  accel compact      : %8.2f ms\n", t_accel_compact );
        std::printf( "  build total        : %8.2f ms\n", t_build_total );
        std::printf( "Mean-shift (OptiX) search: %.2f ms.\n", ms_shift );
        // Per-phase timings summed to a single algorithmic cost, excluding
        // process startup, CUDA/OptiX context init, pipeline/SBT setup, file
        // I/O, and final label assignment. The mean-shift phase includes the
        // BVH build (mandatory acceleration setup for the ray-traced search).
        const double t_meanshift = t_build_total + ms_shift;
        const double t_sum = t_estimate + t_binning + t_meanshift + t_dedup;
        std::printf( "PHASE ms: estimate=%.2f binning=%.2f meanshift=%.2f "
                     "(bvh=%.2f + search=%.2f) modemerge=%.2f | SUM=%.2f\n",
                     t_estimate, t_binning, t_meanshift, t_build_total,
                     (double)ms_shift, t_dedup, t_sum );
        std::printf( "Found %d clusters.\n\n", k );
        std::printf( "%7s  %8s  %14s  %14s\n", "cluster", "count", "center_x", "center_y" );
        for( int oi = 0; oi < k && oi < 50; ++oi )
        {
            const int c = order[oi];
            std::printf( "%7d  %8d  %14.1f  %14.1f\n", c, counts[c], ux[c], uy[c] );
        }
        if( k > 50 ) std::printf( "... (%d more clusters)\n", k - 50 );

        // ---- write per-point labels ----
        std::ofstream out( args.output );
        out << "CASE#,X,Y,cluster_label\n";
        for( int i = 0; i < n; ++i )
            out << cases[i] << ',' << (long long)std::llround( xs[i] ) << ','
                << (long long)std::llround( ys[i] ) << ',' << labels[i] << '\n';
        out.close();
        std::printf( "\nWrote per-point labels to %s\n", args.output.c_str() );

        //
        // Cleanup
        //
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_p ) ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_ids ) ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_done ) ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_cur ) ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_outx ) ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_outy ) ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_count ) ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_conv ) ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_param ) ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.raygenRecord ) ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.missRecordBase ) ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_gas_output_buffer ) ) );

        OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
        OPTIX_CHECK( optixProgramGroupDestroy( hitgroup_prog_group ) );
        OPTIX_CHECK( optixProgramGroupDestroy( miss_prog_group ) );
        OPTIX_CHECK( optixProgramGroupDestroy( raygen_prog_group ) );
        OPTIX_CHECK( optixModuleDestroy( module ) );
        OPTIX_CHECK( optixDeviceContextDestroy( context ) );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
