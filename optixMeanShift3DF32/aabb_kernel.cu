//
// Regular CUDA kernel (compiled to a normal object, NOT OptiX-IR) that builds
// the sphere AABBs on the GPU directly from the device point coordinates, so
// the host never materializes or uploads the AABB array.
//
// Each data point i becomes a sphere of radius = bandwidth; its bounding box is
// [x +/- half_extent]^3 with half_extent = bandwidth + pad (pad absorbs FP32
// rounding in the ray-AABB cull). Points are packed as float4 (B2): one
// 16-byte read per primitive.
//

#include <optix.h>          // OptixAabb
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

__global__ void fill_aabbs_kernel( OptixAabb* __restrict__ out,
                                   const float4* __restrict__ p,
                                   int n, float half_extent )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i >= n ) return;

    const float4 pt = p[i];

    OptixAabb a;
    a.minX = pt.x - half_extent;  a.minY = pt.y - half_extent;  a.minZ = pt.z - half_extent;
    a.maxX = pt.x + half_extent;  a.maxY = pt.y + half_extent;  a.maxZ = pt.z + half_extent;
    out[i] = a;
}

extern "C" void launch_fill_aabbs( OptixAabb* d_aabb, const float4* d_p,
                                   int n, float half_extent, cudaStream_t stream )
{
    const int blk  = 256;
    const int grid = ( n + blk - 1 ) / blk;
    fill_aabbs_kernel<<<grid, blk, 0, stream>>>( d_aabb, d_p, n, half_extent );
}

// ---------------------------------------------------------------------------
// B6: Morton-order reorder of the DEVICE point array (3D: 21 bits per axis
// interleaved into a 63-bit key). Only the device copy is reordered -- window
// sums are order-invariant, so clusters.csv is byte-identical either way.
// ---------------------------------------------------------------------------

// Spread the lower 21 bits of v so there are two zero bits between each.
static __device__ __forceinline__ unsigned long long part1by2_64( unsigned int x )
{
    unsigned long long v = x & 0x1FFFFFull;
    v = ( v | ( v << 32 ) ) & 0x001F00000000FFFFull;
    v = ( v | ( v << 16 ) ) & 0x001F0000FF0000FFull;
    v = ( v | ( v << 8 ) )  & 0x100F00F00F00F00Full;
    v = ( v | ( v << 4 ) )  & 0x10C30C30C30C30C3ull;
    v = ( v | ( v << 2 ) )  & 0x1249249249249249ull;
    return v;
}

__global__ void morton_keys_kernel( const float4* __restrict__ p,
                                    unsigned long long* __restrict__ keys,
                                    int n, float inv_maxr )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i >= n ) return;
    const float4 pt = p[i];
    const float sx = fminf( fmaxf( pt.x * inv_maxr, 0.f ), 1.f );
    const float sy = fminf( fmaxf( pt.y * inv_maxr, 0.f ), 1.f );
    const float sz = fminf( fmaxf( pt.z * inv_maxr, 0.f ), 1.f );
    const unsigned int ux = (unsigned int)( sx * 2097151.f );
    const unsigned int uy = (unsigned int)( sy * 2097151.f );
    const unsigned int uz = (unsigned int)( sz * 2097151.f );
    keys[i] = ( part1by2_64( ux ) << 2 ) | ( part1by2_64( uy ) << 1 ) | part1by2_64( uz );
}

extern "C" void launch_morton_reorder( float4* d_p, int n, float maxr, cudaStream_t stream )
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
