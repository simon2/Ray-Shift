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

// 3D ray-traced mean-shift. Each data point is a sphere of radius = bandwidth;
// each seed shoots a degenerate FLT_MIN point-query ray from the current
// center and the intersection program only tests "origin inside sphere".
// Carries the same B2/B3/B6/C2 optimizations as ../optixMeanShift: points
// packed as float4 (one 16B read, B2) with coordinates handed IS -> AH via
// attribute registers; all per-ray state in payload registers (B3); optional
// Morton reorder of the device copy (B6); host-driven chunked iteration with
// converged-seed compaction (C2).

struct Params
{
    // Recentered data points (device), packed float4 (w unused). Coordinates
    // are integer-valued and small enough after recentering to be exact in
    // float.
    const float4*          p;
    int                    num_points;

    // C2: launch covers slots [0, n_active); slot -> original seed id via ids;
    // moving centers in cur (in/out, w unused); done[slot] = converged.
    const int*             ids;
    double4*               cur;      // DC: center trajectory kept in fp64
    int*                   done;
    int                    n_active;
    int                    chunk;

    // Per-seed outputs, indexed by ORIGINAL seed id (host adds offsets back).
    double*                out_x;
    double*                out_y;
    double*                out_z;
    int*                   out_count;    // density: in-window count at last pre-update center
    int*                   out_conv;     // 1 = converged, 0 = hit the max_iter cap

    double                  h2;           // bandwidth^2 (sphere radius squared)
    double                  stop_thresh2; // (1e-3 * bandwidth)^2

    OptixTraversableHandle handle;
};


struct RayGenData
{
    // No data needed; everything lives in Params.
};


struct MissData
{
    // No data needed.
};


struct HitGroupData
{
    // No data needed; spheres live in Params (primitive index -> point index).
};
