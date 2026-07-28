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

// Ray-traced mean-shift, replacing cuda-ms's mean_shift_kernel. Each data point
// is a flat disk of radius = bandwidth at z = 0; each seed issues a zero-length
// point-query ray on that plane (unified with the 3D variants; a z-spanning
// vertical ray visits the identical node set -- OPTIMIZATIONS.md B11). The
// query at (cx, cy) reaches disk i iff the horizontal
// distance to point i is <= bandwidth -- i.e. point i is inside the flat window.
// The BVH over the disks is the neighbour-query index; an any-hit program sums
// the in-window points and the ray-gen program iterates the center to the mode.

struct Params
{
    // Recentered data points (device), packed as float2 so IS fetches both
    // coordinates in one 8-byte transaction (B2). Coordinates are
    // integer-valued and small enough after recentering to be exact in float.
    const float2*          p;
    int                    num_points;

    // C2: the host drives the iteration in chunks and compacts converged
    // seeds away between launches. A launch covers slots [0, n_active); slot
    // -> original seed id via ids; the moving centers live in cur (in/out);
    // done[slot] reports convergence. Within one launch each slot runs at most
    // `chunk` iterations (chunk == total max_iter reproduces the old
    // single-launch behaviour exactly).
    const int*             ids;
    double2*               cur;      // DC: center trajectory kept in fp64
    int*                   done;
    int                    n_active;
    int                    chunk;

    // Per-seed outputs, indexed by ORIGINAL seed id (host adds offsets back).
    double*                out_x;
    double*                out_y;
    int*                   out_count;    // density: in-window count at last pre-update center
    int*                   out_conv;     // 1 = converged, 0 = hit the max_iter cap

    double                 h2;           // bandwidth^2 (DC: double window test)
    double                 stop_thresh2; // (1e-3 * bandwidth)^2 (double)

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
    // No data needed; disks live in Params (primitive index -> point index).
};
