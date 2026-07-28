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

#include <optix.h>

#include "optixMeanShift3DF32MG.h"

#include <sutil/vec_math.h>

#include <cfloat>   // FLT_MIN: shortest positive ray length OptiX accepts

extern "C" {
__constant__ Params params;
}

// Per-ray state lives entirely in payload registers (B3), no PRD struct:
//   payload 0, 1, 2 : current center cx, cy, cz (float bits)
//   payload 3       : sx float accumulator (float bits)
//   payload 4       : sy float accumulator (float bits)
//   payload 5       : sz float accumulator (float bits)
//   payload 6       : in-window count
// The intersection program hands the accepted point's coordinates to the
// any-hit program through attribute registers (B2), so the point data is read
// from global memory exactly once per candidate. Pure float32 variant:
// storage, window test, accumulation and center updates are all fp32.


// One thread per ACTIVE seed slot (C2). Runs up to params.chunk iterations of
// the slot's trajectory: each iteration shoots a degenerate point-query ray at
// the current center, reads back the in-window sums/count from the payload
// registers, and shifts the center to the window mean. State is persisted to
// cur/done and the (original-id-indexed) out arrays after every chunk so the
// host can compact converged seeds away between launches. chunk == max_iter
// reproduces single-launch behaviour exactly.
extern "C" __global__ void __raygen__rg()
{
    const unsigned int s = optixGetLaunchIndex().x;
    if( s >= static_cast<unsigned int>( params.n_active ) )
        return;

    const int    orig = params.ids[s];
    const float4 c0   = params.cur[s];
    float cx = c0.x;
    float cy = c0.y;
    float cz = c0.z;
    int   last_n = 0;   // density at the pre-update center
    int   conv   = 0;

    for( int it = 0; it < params.chunk; ++it )
    {
        unsigned int u0 = __float_as_uint( cx );
        unsigned int u1 = __float_as_uint( cy );
        unsigned int u2 = __float_as_uint( cz );
        unsigned int u3 = __float_as_uint( 0.0f );
        unsigned int u4 = __float_as_uint( 0.0f );
        unsigned int u5 = __float_as_uint( 0.0f );
        unsigned int u6 = 0;

        // Point query: any sphere whose AABB contains the origin triggers the IS
        // program. The ulp pad keeps the hardware AABB cull conservative; the
        // in-window test is done in the IS program.
        optixTrace(
                params.handle,
                make_float3( cx, cy, cz ),
                make_float3( 0.0f, 0.0f, 1.0f ),
                0.0f,                       // tmin
                FLT_MIN,                    // tmax
                0.0f,                       // rayTime
                OptixVisibilityMask( 1 ),
                OPTIX_RAY_FLAG_NONE,        // keep traversing: any-hit enumerates all spheres
                0,                          // SBT offset
                0,                          // SBT stride
                0,                          // missSBTIndex
                u0, u1, u2, u3, u4, u5, u6 );

        const int n = (int)u6;
        last_n = n;
        if( n == 0 )
        {
            conv = 1;   // empty window: stationary seed (matches cuda-ms)
            break;
        }

        const float sx = __uint_as_float( u3 );
        const float sy = __uint_as_float( u4 );
        const float sz = __uint_as_float( u5 );

        // IEEE float division (__fdiv_rn): the module is compiled with
        // --use_fast_math, which would otherwise turn / into div.approx.
        const float nx  = __fdiv_rn( sx, (float)n );
        const float ny  = __fdiv_rn( sy, (float)n );
        const float nz  = __fdiv_rn( sz, (float)n );
        const float ddx = nx - cx;
        const float ddy = ny - cy;
        const float ddz = nz - cz;

        cx = nx;
        cy = ny;
        cz = nz;

        if( ddx * ddx + ddy * ddy + ddz * ddz < params.stop_thresh2 )
        {
            conv = 1;
            break;
        }
    }

    params.cur[s]          = make_float4( cx, cy, cz, 0.f );
    params.done[s]         = conv;
    params.out_x[orig]     = cx;
    params.out_y[orig]     = cy;
    params.out_z[orig]     = cz;
    params.out_count[orig] = last_n;
    params.out_conv[orig]  = conv;
}


extern "C" __global__ void __miss__ms()
{
    // All intersections are ignored, so every ray ends here. Nothing to do; the
    // accumulation already happened in the any-hit program.
}


// Custom primitive = sphere of radius = bandwidth. Reaching this program means
// the query origin is inside the sphere's AABB; do the exact inside test. The
// center is read from payload registers (B3); on accept the point's
// coordinates ride to the any-hit program in attribute registers (B2) -- one
// 16-byte global read per candidate, total.
extern "C" __global__ void __intersection__sphere()
{
    const unsigned int i = optixGetPrimitiveIndex();

    const float  cx = __uint_as_float( optixGetPayload_0() );
    const float  cy = __uint_as_float( optixGetPayload_1() );
    const float  cz = __uint_as_float( optixGetPayload_2() );
    const float4 pt = params.p[i];

    const float dx = cx - pt.x;
    const float dy = cy - pt.y;
    const float dz = cz - pt.z;

    if( dx * dx + dy * dy + dz * dz <= params.h2 )
        optixReportIntersection( 0.0f, 0,                     // t = tmin; any-hit ignores it
                                 __float_as_uint( pt.x ),
                                 __float_as_uint( pt.y ),
                                 __float_as_uint( pt.z ) );
}


// Accumulate the in-window point into the payload-register sums (int64 split
// across two registers each, exact) and let traversal continue so every
// containing sphere is visited. Coordinates arrive via attribute registers;
// no memory access at all in this program.
extern "C" __global__ void __anyhit__accumulate()
{
    const float x = __uint_as_float( optixGetAttribute_0() );
    const float y = __uint_as_float( optixGetAttribute_1() );
    const float z = __uint_as_float( optixGetAttribute_2() );

    optixSetPayload_3( __float_as_uint( __uint_as_float( optixGetPayload_3() ) + x ) );
    optixSetPayload_4( __float_as_uint( __uint_as_float( optixGetPayload_4() ) + y ) );
    optixSetPayload_5( __float_as_uint( __uint_as_float( optixGetPayload_5() ) + z ) );
    optixSetPayload_6( optixGetPayload_6() + 1u );

    optixIgnoreIntersection();
}
