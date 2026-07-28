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

#include "optixMeanShiftF32MG.h"

#include <sutil/vec_math.h>

#include <cfloat>   // FLT_MIN: shortest positive ray length OptiX accepts

extern "C" {
__constant__ Params params;
}

// Per-ray state lives entirely in payload registers (B3), no PRD struct:
//   payload 0, 1 : current center cx, cy (float bits)
//   payload 2    : sx float accumulator (float bits)
//   payload 3    : sy float accumulator (float bits)
//   payload 4    : in-window count
// The intersection program hands the accepted point's coordinates to the
// any-hit program through attribute registers (B2), so the point data is read
// from global memory exactly once per candidate. Pure float32 variant:
// storage, window test, accumulation and center updates are all fp32.


// One thread per ACTIVE seed slot (C2). Runs up to params.chunk iterations of
// the slot's trajectory: each iteration shoots a vertical ray at the current
// center, reads back the in-window sums/count from the payload registers, and
// shifts the center to the window mean. State is persisted to cur/done and the
// (original-id-indexed) out arrays after every chunk so the host can compact
// converged seeds away between launches. chunk == max_iter reproduces the old
// single-launch behaviour exactly. Mirrors cuda-ms's mean_shift_kernel.
extern "C" __global__ void __raygen__rg()
{
    const unsigned int s = optixGetLaunchIndex().x;
    if( s >= static_cast<unsigned int>( params.n_active ) )
        return;

    const int    orig = params.ids[s];
    const float2 c0   = params.cur[s];
    float cx = c0.x;
    float cy = c0.y;
    int   last_n = 0;   // density at the pre-update center
    int   conv   = 0;

    for( int it = 0; it < params.chunk; ++it )
    {
        unsigned int u0 = __float_as_uint( cx );
        unsigned int u1 = __float_as_uint( cy );
        unsigned int u2 = __float_as_uint( 0.0f );
        unsigned int u3 = __float_as_uint( 0.0f );
        unsigned int u4 = 0;

        // Point query (unified with the 3D variants): zero-length ray sitting
        // ON the z=0 disk plane -- visits exactly the leaves whose AABB contains
        // (cx, cy); same node set and cost as a z-spanning vertical ray
        // (measured A/B, ../optixMeanShift/OPTIMIZATIONS.md B11).
        optixTrace(
                params.handle,
                make_float3( cx, cy, 0.0f ),
                make_float3( 0.0f, 0.0f, 1.0f ),
                0.0f,                       // tmin
                FLT_MIN,                    // tmax
                0.0f,                       // rayTime
                OptixVisibilityMask( 1 ),
                OPTIX_RAY_FLAG_NONE,        // keep traversing: any-hit enumerates all disks
                0,                          // SBT offset
                0,                          // SBT stride
                0,                          // missSBTIndex
                u0, u1, u2, u3, u4 );

        const int n = (int)u4;
        last_n = n;
        if( n == 0 )
        {
            conv = 1;   // empty window: stationary seed (matches cuda-ms)
            break;
        }

        const float sx = __uint_as_float( u2 );
        const float sy = __uint_as_float( u3 );

        // IEEE float division (__fdiv_rn): the module is compiled with
        // --use_fast_math, which would otherwise turn / into div.approx.
        const float nx  = __fdiv_rn( sx, (float)n );
        const float ny  = __fdiv_rn( sy, (float)n );
        const float ddx = nx - cx;
        const float ddy = ny - cy;

        cx = nx;
        cy = ny;

        if( ddx * ddx + ddy * ddy < params.stop_thresh2 )
        {
            conv = 1;
            break;
        }
    }

    params.cur[s]          = make_float2( cx, cy );
    params.done[s]         = conv;
    params.out_x[orig]     = cx;
    params.out_y[orig]     = cy;
    params.out_count[orig] = last_n;
    params.out_conv[orig]  = conv;
}


extern "C" __global__ void __miss__ms()
{
    // All intersections are ignored, so every ray ends here. Nothing to do; the
    // accumulation already happened in the any-hit program.
}


// Custom primitive = flat disk of radius = bandwidth at z=0. For the vertical
// ray this collapses to a horizontal in-window test. Points are integer-valued
// (exact in float); the center is read from payload registers (B3). On accept,
// the point's coordinates ride to the any-hit program in attribute registers
// (B2) -- one 8-byte global read per candidate, total.
extern "C" __global__ void __intersection__disk()
{
    const unsigned int i = optixGetPrimitiveIndex();

    const float  cx = __uint_as_float( optixGetPayload_0() );
    const float  cy = __uint_as_float( optixGetPayload_1() );
    const float2 pt = params.p[i];

    const float dx = cx - pt.x;
    const float dy = cy - pt.y;

    if( dx * dx + dy * dy <= params.h2 )
        optixReportIntersection( 0.0f, 0,                     // t = tmin (zero-length ray)
                                 __float_as_uint( pt.x ),
                                 __float_as_uint( pt.y ) );
}


// Accumulate the in-window point into the payload-register sums (int64 split
// across two registers each, exact) and let traversal continue so every
// pierced disk is visited. Coordinates arrive via attribute registers; no
// memory access at all in this program.
extern "C" __global__ void __anyhit__accumulate()
{
    const float x = __uint_as_float( optixGetAttribute_0() );
    const float y = __uint_as_float( optixGetAttribute_1() );

    optixSetPayload_2( __float_as_uint( __uint_as_float( optixGetPayload_2() ) + x ) );
    optixSetPayload_3( __float_as_uint( __uint_as_float( optixGetPayload_3() ) + y ) );
    optixSetPayload_4( optixGetPayload_4() + 1u );

    optixIgnoreIntersection();
}
