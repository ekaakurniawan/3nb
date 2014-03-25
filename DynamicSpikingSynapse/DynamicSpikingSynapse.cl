// Copyright (C) 2014 by Eka A. Kurniawan
// eka.a.kurniawan(ta)gmail(tod)com
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; version 2 of the License.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the
// Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

__kernel void advance_spikeHit_pn_pa(const float t,
                                     const int step,
                                     const int nSteps,
                                     __global const int *hasFireds,
                                     __global const int *ttl_incoming_synapses,
                                     __global const long *incoming_synapse_start_idxs,
                                     __global float *dss_PSR,
                                     __global const float *dss_W,
                                     __global const float *dss_decay,
                                     __global const float *dss_U,
                                     __global const float *dss_D,
                                     __global const float *dss_F,
                                     __global float *dss_u,
                                     __global float *dss_r,
                                     __global float *dss_lastSpike,
                                     __global float *ttl_PSRs_coll)
{
    long gid = get_global_id(0);
    // Index to store total PSR
    long out_idx = gid * nSteps;
    
    float ttl_PSR = 0;
    int ttl_incoming_synapse = ttl_incoming_synapses[gid];
    long start_idx = incoming_synapse_start_idxs[gid];
    for (int i = 0; i < ttl_incoming_synapse; i++) {
        // Synapse ID
        long sid = start_idx + (long)i;
        // advance
        float PSR = dss_PSR[sid];
        float decay = dss_decay[sid];
        PSR *= decay;
        // preSpikeHit
        if (hasFireds[step]) {
            float r = dss_r[sid];
            float u = dss_u[sid];
            if (dss_lastSpike[sid] > 0) {
                float isi = t - dss_lastSpike[sid];
                r = 1 + \
                    ( ((r * (1 - u)) - 1) * \
                      exp(-isi / dss_D[sid]) );
                float U = dss_U[sid];
                u = U + \
                    ( u  * (1 - U) * exp(-isi / dss_F[sid]) );
                dss_r[sid] = r;
                dss_u[sid] = u;
            }
            PSR += ((dss_W[sid] / decay) * u * r);
            dss_lastSpike[sid] = t;
        }
        dss_PSR[sid] = PSR;
        ttl_PSR += PSR;
    }
    ttl_PSRs_coll[out_idx + step] = ttl_PSR;
}

__kernel void advance_spikeHit_ps_pa(const float t,
                                     const int step,
                                     __global const int *hasFireds,
                                     __global float *dss_PSR,
                                     __global const float *dss_W,
                                     __global const float *dss_decay,
                                     __global const float *dss_U,
                                     __global const float *dss_D,
                                     __global const float *dss_F,
                                     __global float *dss_u,
                                     __global float *dss_r,
                                     __global float *dss_lastSpike)
{
    long gid = get_global_id(0);
    
    // advance
    float PSR = dss_PSR[gid];
    float decay = dss_decay[gid];
    PSR *= decay;
    // preSpikeHit
    if (hasFireds[step]) {
        float r = dss_r[gid];
        float u = dss_u[gid];
        if (dss_lastSpike[gid] > 0) {
            float isi = t - dss_lastSpike[gid];
            r = 1 + \
                ( ((r * (1 - u)) - 1) * \
                  exp(-isi / dss_D[gid]) );
            float U = dss_U[gid];
            u = U + \
                ( u  * (1 - U) * exp(-isi / dss_F[gid]) );
            dss_r[gid] = r;
            dss_u[gid] = u;
        }
        PSR += ((dss_W[gid] / decay) * u * r);
        dss_lastSpike[gid] = t;
    }
    dss_PSR[gid] = PSR;
}

__kernel void sum_up_psr(const int step,
                         const int nSteps,
                         __global const int *ttl_incoming_synapses,
                         __global const long *incoming_synapse_start_idxs,
                         __global float *dss_PSR,
                         __global float *ttl_PSRs_coll)
{
    long gid = get_global_id(0);
    // Index to store total PSR
    long out_idx = gid * nSteps;
    
    float ttl_PSR = 0;
    int ttl_incoming_synapse = ttl_incoming_synapses[gid];
    long start_idx = incoming_synapse_start_idxs[gid];
    for (int i = 0; i < ttl_incoming_synapse; i++) {
        // Synapse ID
        long sid = start_idx + (long)i;
        // Sum up PSR
        float PSR = dss_PSR[sid];
        ttl_PSR += PSR;
    }
    ttl_PSRs_coll[out_idx + step] = ttl_PSR;
}