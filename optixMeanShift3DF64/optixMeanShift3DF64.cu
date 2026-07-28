//
// optixMeanShift3DF64: DOUBLE-ACCUMULATION variant of ../optixMeanShift3DDC.
// Identical to 3D DC (fp64 center, +1 conservative AABB pad, double window test
// against the payload-carried true center, one double division per iteration)
// EXCEPT the window sum is accumulated in fp64 instead of exact int64, matching
// the hybrid F64 arm so the FP64-tax comparison is apples-to-apples. For integer
// coords with running sum < 2^53 the double sum equals the int64 sum bit-for-bit,
// so labels are BYTE-IDENTICAL to 3D DC; only timing differs. See
// ../optixMeanShiftF64 for the full rationale.
//
// Payload (13 registers):
//   payload 0, 1   : cx double (lo, hi bits)
//   payload 2, 3   : cy double (lo, hi bits)
//   payload 4, 5   : cz double (lo, hi bits)
//   payload 6, 7   : sx double accumulator (lo, hi bits)
//   payload 8, 9   : sy double accumulator (lo, hi bits)
//   payload 10, 11 : sz double accumulator (lo, hi bits)
//   payload 12     : in-window count
//

#include <optix.h>

#include "optixMeanShift3DF64.h"

#include <sutil/vec_math.h>

#include <cfloat>   // FLT_MIN

extern "C" {
__constant__ Params params;
}

static __forceinline__ __device__ unsigned int dlo( double v )
{
    return (unsigned int)( (unsigned long long)__double_as_longlong( v ) & 0xffffffffull );
}
static __forceinline__ __device__ unsigned int dhi( double v )
{
    return (unsigned int)( (unsigned long long)__double_as_longlong( v ) >> 32 );
}
static __forceinline__ __device__ double djoin( unsigned int lo, unsigned int hi )
{
    return __longlong_as_double( (long long)( ( (unsigned long long)hi << 32 ) | lo ) );
}

// fp64 accumulator carried as two payload registers (lo/hi bits of the running
// sum). Sequential any-hit within a ray makes the read-modify-write safe.
#define DADD_PAYLOAD( LO, HI, NAME )                                           \
static __forceinline__ __device__ void NAME( double add )                      \
{                                                                              \
    const double s = djoin( optixGetPayload_##LO(), optixGetPayload_##HI() ) + add; \
    optixSetPayload_##LO( dlo( s ) );                                          \
    optixSetPayload_##HI( dhi( s ) );                                          \
}

DADD_PAYLOAD( 6, 7, dadd_sx )
DADD_PAYLOAD( 8, 9, dadd_sy )
DADD_PAYLOAD( 10, 11, dadd_sz )


extern "C" __global__ void __raygen__rg()
{
    const unsigned int s = optixGetLaunchIndex().x;
    if( s >= static_cast<unsigned int>( params.n_active ) )
        return;

    const int     orig = params.ids[s];
    const double4 c0   = params.cur[s];
    double cx = c0.x;
    double cy = c0.y;
    double cz = c0.z;
    int    last_n = 0;
    int    conv   = 0;

    for( int it = 0; it < params.chunk; ++it )
    {
        unsigned int u0 = dlo( cx ), u1 = dhi( cx );
        unsigned int u2 = dlo( cy ), u3 = dhi( cy );
        unsigned int u4 = dlo( cz ), u5 = dhi( cz );
        unsigned int u6 = 0, u7 = 0, u8 = 0, u9 = 0, u10 = 0, u11 = 0, u12 = 0;

        optixTrace(
                params.handle,
                make_float3( (float)cx, (float)cy, (float)cz ),
                make_float3( 0.0f, 0.0f, 1.0f ),
                0.0f,                       // tmin
                FLT_MIN,                    // tmax
                0.0f,                       // rayTime
                OptixVisibilityMask( 1 ),
                OPTIX_RAY_FLAG_NONE,
                0, 0, 0,
                u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12 );

        const int n = (int)u12;
        last_n = n;
        if( n == 0 )
        {
            conv = 1;
            break;
        }

        const double sx = djoin( u6, u7 );
        const double sy = djoin( u8, u9 );
        const double sz = djoin( u10, u11 );

        const double nx = sx / static_cast<double>( n );
        const double ny = sy / static_cast<double>( n );
        const double nz = sz / static_cast<double>( n );

        const double ddx = nx - cx;
        const double ddy = ny - cy;
        const double ddz = nz - cz;

        cx = nx;
        cy = ny;
        cz = nz;

        if( ddx * ddx + ddy * ddy + ddz * ddz < params.stop_thresh2 )
        {
            conv = 1;
            break;
        }
    }

    params.cur[s]          = make_double4( cx, cy, cz, 0.0 );
    params.done[s]         = conv;
    params.out_x[orig]     = cx;
    params.out_y[orig]     = cy;
    params.out_z[orig]     = cz;
    params.out_count[orig] = last_n;
    params.out_conv[orig]  = conv;
}


extern "C" __global__ void __miss__ms()
{
}


extern "C" __global__ void __intersection__sphere()
{
    const unsigned int i = optixGetPrimitiveIndex();

    const double cx = djoin( optixGetPayload_0(), optixGetPayload_1() );
    const double cy = djoin( optixGetPayload_2(), optixGetPayload_3() );
    const double cz = djoin( optixGetPayload_4(), optixGetPayload_5() );
    const float4 pt = params.p[i];

    const double dx = cx - (double)pt.x;
    const double dy = cy - (double)pt.y;
    const double dz = cz - (double)pt.z;

    if( dx * dx + dy * dy + dz * dz <= params.h2 )
        optixReportIntersection( 0.0f, 0,
                                 __float_as_uint( pt.x ),
                                 __float_as_uint( pt.y ),
                                 __float_as_uint( pt.z ) );
}


extern "C" __global__ void __anyhit__accumulate()
{
    const float x = __uint_as_float( optixGetAttribute_0() );
    const float y = __uint_as_float( optixGetAttribute_1() );
    const float z = __uint_as_float( optixGetAttribute_2() );

    dadd_sx( (double)x );
    dadd_sy( (double)y );
    dadd_sz( (double)z );
    optixSetPayload_12( optixGetPayload_12() + 1u );

    optixIgnoreIntersection();
}
