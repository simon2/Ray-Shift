//
// optixMeanShiftF64 (2D): DOUBLE-ACCUMULATION variant of ../optixMeanShiftDC.
//
// Identical to DC in every respect (double center trajectory, +1-unit
// conservative AABB pad, double in-window test against the true payload
// center, one correctly-rounded double division per iteration) EXCEPT the
// window sum is accumulated in fp64 instead of exact int64. Purpose: give the
// FP64 comparison a consistent "everything in double" definition, matching the
// hybrid F64 arm (../cuda-ms-fp64), so the FP64-tax numbers are apples-to-apples
// (no arm dodging the gimped fp64 units via an integer accumulator).
//
// Correctness: for integer coordinates with the running sum < 2^53 (holds for
// every dataset here: <= 2^24.5 pts x <= 2^24 coord = 2^48.5), each fp64 add of
// an integer to an integer sum <= 2^53 is exact, so the double sum equals the
// int64 sum bit-for-bit. Output labels are therefore BYTE-IDENTICAL to DC; only
// the timing (fp64 accumulate now on the hot path) differs.
//
// Payload (9 registers, 2D):
//   payload 0, 1 : cx double (lo, hi bits)
//   payload 2, 3 : cy double (lo, hi bits)
//   payload 4, 5 : sx double accumulator (lo, hi bits)
//   payload 6, 7 : sy double accumulator (lo, hi bits)
//   payload 8    : in-window count
//

#include <optix.h>

#include "optixMeanShiftF64.h"

#include <sutil/vec_math.h>

#include <cfloat>   // FLT_MIN: shortest positive ray length OptiX accepts

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

// Double accumulator carried as two payload registers (lo/hi bits of the fp64
// running sum). Read-modify-write is safe: within one ray the any-hit calls run
// sequentially, so this mirrors DC's int64 add64 but in fp64.
static __forceinline__ __device__ void dadd_payload_4_5( double add )
{
    const double s = djoin( optixGetPayload_4(), optixGetPayload_5() ) + add;
    optixSetPayload_4( dlo( s ) );
    optixSetPayload_5( dhi( s ) );
}

static __forceinline__ __device__ void dadd_payload_6_7( double add )
{
    const double s = djoin( optixGetPayload_6(), optixGetPayload_7() ) + add;
    optixSetPayload_6( dlo( s ) );
    optixSetPayload_7( dhi( s ) );
}


extern "C" __global__ void __raygen__rg()
{
    const unsigned int s = optixGetLaunchIndex().x;
    if( s >= static_cast<unsigned int>( params.n_active ) )
        return;

    const int     orig = params.ids[s];
    const double2 c0   = params.cur[s];
    double cx = c0.x;
    double cy = c0.y;
    int    last_n = 0;   // density at the pre-update center
    int    conv   = 0;

    for( int it = 0; it < params.chunk; ++it )
    {
        unsigned int u0 = dlo( cx ), u1 = dhi( cx );
        unsigned int u2 = dlo( cy ), u3 = dhi( cy );
        unsigned int u4 = 0, u5 = 0, u6 = 0, u7 = 0, u8 = 0;

        // Point query launched from the FLOAT image of the double center; the
        // +1-unit AABB pad guarantees no true window member is culled.
        optixTrace(
                params.handle,
                make_float3( (float)cx, (float)cy, 0.0f ),
                make_float3( 0.0f, 0.0f, 1.0f ),
                0.0f,                       // tmin
                FLT_MIN,                    // tmax
                0.0f,                       // rayTime
                OptixVisibilityMask( 1 ),
                OPTIX_RAY_FLAG_NONE,        // keep traversing: any-hit enumerates all disks
                0,                          // SBT offset
                0,                          // SBT stride
                0,                          // missSBTIndex
                u0, u1, u2, u3, u4, u5, u6, u7, u8 );

        const int n = (int)u8;
        last_n = n;
        if( n == 0 )
        {
            conv = 1;   // empty window: stationary seed (matches cuda-ms)
            break;
        }

        // fp64-accumulated sums (bit-identical to DC's int64 sum for integer
        // coords, running sum < 2^53); one correctly-rounded double division,
        // center STAYS double.
        const double sx = djoin( u4, u5 );
        const double sy = djoin( u6, u7 );

        const double nx = sx / static_cast<double>( n );
        const double ny = sy / static_cast<double>( n );

        const double ddx = nx - cx;
        const double ddy = ny - cy;

        cx = nx;
        cy = ny;

        if( ddx * ddx + ddy * ddy < params.stop_thresh2 )
        {
            conv = 1;
            break;
        }
    }

    params.cur[s]          = make_double2( cx, cy );
    params.done[s]         = conv;
    params.out_x[orig]     = cx;
    params.out_y[orig]     = cy;
    params.out_count[orig] = last_n;
    params.out_conv[orig]  = conv;
}


extern "C" __global__ void __miss__ms()
{
    // All intersections are ignored, so every ray ends here.
}


// The in-window decision is made against the TRUE double center from the
// payload; the fp32 ray origin only steered the (padded, conservative)
// traversal. Points are integers (exact in both float and double).
extern "C" __global__ void __intersection__disk()
{
    const unsigned int i = optixGetPrimitiveIndex();

    const double cx = djoin( optixGetPayload_0(), optixGetPayload_1() );
    const double cy = djoin( optixGetPayload_2(), optixGetPayload_3() );
    const float2 pt = params.p[i];

    const double dx = cx - (double)pt.x;
    const double dy = cy - (double)pt.y;

    if( dx * dx + dy * dy <= params.h2 )
        optixReportIntersection( 0.0f, 0,                     // t = tmin (zero-length ray)
                                 (unsigned int)(long long)pt.x,
                                 (unsigned int)(long long)pt.y );
}


extern "C" __global__ void __anyhit__accumulate()
{
    dadd_payload_4_5( (double)optixGetAttribute_0() );
    dadd_payload_6_7( (double)optixGetAttribute_1() );
    optixSetPayload_8( optixGetPayload_8() + 1u );

    optixIgnoreIntersection();
}
