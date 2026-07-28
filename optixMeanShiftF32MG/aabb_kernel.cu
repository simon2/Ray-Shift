//
// Regular CUDA kernel (compiled to a normal object, NOT OptiX-IR) that builds
// the disk AABBs on the GPU directly from the device point coordinates, so the
// host never materializes or uploads the AABB array.
//
// Each data point i becomes a flat disk of radius = bandwidth at z = 0; its
// bounding box is [x +/- half_extent] x [y +/- half_extent] x [0, 0], where
// half_extent = bandwidth exactly (no pad: with integer coords and an integer
// bandwidth the box endpoints are float-exact, so the hardware cull's per-axis
// comparisons are exact). Points are packed as float2 (B2): one 8-byte read
// per primitive.
//

#include <optix.h>          // OptixAabb
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

__global__ void fill_aabbs_kernel( OptixAabb* __restrict__ out,
                                   const float2* __restrict__ p,
                                   int n, float half_extent )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i >= n ) return;

    const float2 pt = p[i];

    OptixAabb a;
    a.minX = pt.x - half_extent;  a.minY = pt.y - half_extent;  a.minZ = 0.0f;
    a.maxX = pt.x + half_extent;  a.maxY = pt.y + half_extent;  a.maxZ = 0.0f;
    out[i] = a;
}

extern "C" void launch_fill_aabbs( OptixAabb* d_aabb, const float2* d_p,
                                   int n, float half_extent, cudaStream_t stream )
{
    const int blk  = 256;
    const int grid = ( n + blk - 1 ) / blk;
    fill_aabbs_kernel<<<grid, blk, 0, stream>>>( d_aabb, d_p, n, half_extent );
}

// ---------------------------------------------------------------------------
// B6: Morton-order reorder of the DEVICE point array. The BVH builder groups
// spatially nearby primitives into neighbouring leaves; feeding it points that
// are already in Morton order makes "nearby in the tree" == "nearby in the
// p[] array", so the scattered p[primIndex] reads of the IS program become
// cache-friendly. Only the device copy is reordered -- the host keeps the
// original order, and window sums are order-invariant, so clusters.csv is
// byte-identical with or without the reorder.
// ---------------------------------------------------------------------------

// Interleave the lower 32 bits of v with zeros (Morton "part1by1").
static __device__ __forceinline__ unsigned long long part1by1_64( unsigned int x )
{
    unsigned long long v = x;
    v = ( v | ( v << 16 ) ) & 0x0000FFFF0000FFFFull;
    v = ( v | ( v << 8 ) )  & 0x00FF00FF00FF00FFull;
    v = ( v | ( v << 4 ) )  & 0x0F0F0F0F0F0F0F0Full;
    v = ( v | ( v << 2 ) )  & 0x3333333333333333ull;
    v = ( v | ( v << 1 ) )  & 0x5555555555555555ull;
    return v;
}

__global__ void morton_keys_kernel( const float2* __restrict__ p,
                                    unsigned long long* __restrict__ keys,
                                    int n, float inv_maxr )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i >= n ) return;
    const float2 pt = p[i];
    // Quantize to a 21-bit grid per axis (locality granularity only; the grid
    // resolution has no effect on results).
    const float sx = fminf( fmaxf( pt.x * inv_maxr, 0.f ), 1.f );
    const float sy = fminf( fmaxf( pt.y * inv_maxr, 0.f ), 1.f );
    const unsigned int ux = (unsigned int)( sx * 2097151.f );
    const unsigned int uy = (unsigned int)( sy * 2097151.f );
    keys[i] = ( part1by1_64( ux ) << 1 ) | part1by1_64( uy );
}

extern "C" void launch_morton_reorder( float2* d_p, int n, float maxr, cudaStream_t stream )
{
    unsigned long long* d_keys = nullptr;
    cudaMalloc( reinterpret_cast<void**>( &d_keys ), (size_t)n * sizeof( unsigned long long ) );
    const int blk  = 256;
    const int grid = ( n + blk - 1 ) / blk;
    morton_keys_kernel<<<grid, blk, 0, stream>>>( d_p, d_keys, n, maxr > 0.f ? 1.f / maxr : 0.f );
    thrust::sort_by_key( thrust::cuda::par.on( stream ),
                         thrust::device_pointer_cast( d_keys ),
                         thrust::device_pointer_cast( d_keys + n ),
                         thrust::device_pointer_cast( d_p ) );
    cudaStreamSynchronize( stream );
    cudaFree( d_keys );
}
