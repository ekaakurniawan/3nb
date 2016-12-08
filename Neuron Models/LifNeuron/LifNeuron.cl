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

// LifNeuron
#define LN_LEN                  13
#define LN_RM_IDX               0
#define LN_V_RESTING_IDX        1
#define LN_V_THRESH_IDX         2
#define LN_V_RESET_IDX          3
#define LN_V_INIT_IDX           4
#define LN_VM_IDX               5
#define LN_T_REFRACT_IDX        6
#define LN_SUMMATION_POINT_IDX  7
#define LN_I_INJECT_IDX         8
#define LN_C1_IDX               9
#define LN_C2_IDX               10
#define LN_N_STEPS_IN_REFR_IDX  11
#define LN_HAS_FIRED_IDX        12


__kernel void nextState_plir(const float dt,
                             const int step,
                             __global const float *Iinjects,
                             __global float *lifNeurons,
                             __global float *Vms,
                             __global int *spikes)
{
    long gid = get_global_id(0);
    long ln_idx = gid * LN_LEN;
    
    lifNeurons[ln_idx + LN_I_INJECT_IDX] = Iinjects[step];
    
    if (lifNeurons[ln_idx + LN_N_STEPS_IN_REFR_IDX] > 0) {
        lifNeurons[ln_idx + LN_N_STEPS_IN_REFR_IDX] -= 1;
        lifNeurons[ln_idx + LN_HAS_FIRED_IDX] = 0;
    } else if (lifNeurons[ln_idx + LN_VM_IDX] >= lifNeurons[ln_idx + LN_V_THRESH_IDX]) {
        lifNeurons[ln_idx + LN_HAS_FIRED_IDX] = 1;
        lifNeurons[ln_idx + LN_N_STEPS_IN_REFR_IDX] =
            (float)ceil(lifNeurons[ln_idx + LN_T_REFRACT_IDX] / dt);
        lifNeurons[ln_idx + LN_VM_IDX] = lifNeurons[ln_idx + LN_V_RESET_IDX];
    } else {
        lifNeurons[ln_idx + LN_HAS_FIRED_IDX] = 0;
        lifNeurons[ln_idx + LN_SUMMATION_POINT_IDX] =
            lifNeurons[ln_idx + LN_SUMMATION_POINT_IDX] +
            lifNeurons[ln_idx + LN_I_INJECT_IDX] +
            (lifNeurons[ln_idx + LN_V_RESTING_IDX] / lifNeurons[ln_idx + LN_RM_IDX]);
        lifNeurons[ln_idx + LN_VM_IDX] =
            (lifNeurons[ln_idx + LN_C1_IDX] * lifNeurons[ln_idx + LN_VM_IDX]) +
            (lifNeurons[ln_idx + LN_C2_IDX] * lifNeurons[ln_idx + LN_SUMMATION_POINT_IDX]);
    }
        
    lifNeurons[ln_idx + LN_SUMMATION_POINT_IDX] = 0;
        
    Vms[gid] = lifNeurons[ln_idx + LN_VM_IDX];
    if (lifNeurons[ln_idx + LN_HAS_FIRED_IDX]) {
        spikes[gid] = 1;
    } else {
        spikes[gid] = 0;
    }
}

__kernel void nextState_plcr(const float dt,
                             const int step,
                             const int input_len,
                             __global const float *Iinjects,
                             __global float *lifNeurons,
                             __global float *Vms,
                             __global int *spikes)
{
    long gid = get_global_id(0);
    // Start index of neuron
    long ln_idx = gid * LN_LEN;
    // Index to store Vm and spike
    long out_idx = gid * input_len;
    
    lifNeurons[ln_idx + LN_I_INJECT_IDX] = Iinjects[step];
    
    if (lifNeurons[ln_idx + LN_N_STEPS_IN_REFR_IDX] > 0) {
        lifNeurons[ln_idx + LN_N_STEPS_IN_REFR_IDX] -= 1;
        lifNeurons[ln_idx + LN_HAS_FIRED_IDX] = 0;
    } else if (lifNeurons[ln_idx + LN_VM_IDX] >= lifNeurons[ln_idx + LN_V_THRESH_IDX]) {
        lifNeurons[ln_idx + LN_HAS_FIRED_IDX] = 1;
        lifNeurons[ln_idx + LN_N_STEPS_IN_REFR_IDX] =
            (float)ceil(lifNeurons[ln_idx + LN_T_REFRACT_IDX] / dt);
        lifNeurons[ln_idx + LN_VM_IDX] = lifNeurons[ln_idx + LN_V_RESET_IDX];
    } else {
        lifNeurons[ln_idx + LN_HAS_FIRED_IDX] = 0;
        lifNeurons[ln_idx + LN_SUMMATION_POINT_IDX] =
            lifNeurons[ln_idx + LN_SUMMATION_POINT_IDX] +
            lifNeurons[ln_idx + LN_I_INJECT_IDX] +
            (lifNeurons[ln_idx + LN_V_RESTING_IDX] / lifNeurons[ln_idx + LN_RM_IDX]);
        lifNeurons[ln_idx + LN_VM_IDX] =
            (lifNeurons[ln_idx + LN_C1_IDX] * lifNeurons[ln_idx + LN_VM_IDX]) +
            (lifNeurons[ln_idx + LN_C2_IDX] * lifNeurons[ln_idx + LN_SUMMATION_POINT_IDX]);
    }
    
    lifNeurons[ln_idx + LN_SUMMATION_POINT_IDX] = 0;
    
    Vms[out_idx + step] = lifNeurons[ln_idx + LN_VM_IDX];
    if (lifNeurons[ln_idx + LN_HAS_FIRED_IDX]) {
        spikes[out_idx + step] = 1;
    } else {
        spikes[out_idx + step] = 0;
    }
}

__kernel void nextState_clcr(const float dt,
                             const int input_len,
                             __global const float *Iinjects,
                             __global float *lifNeurons,
                             __global float *Vms,
                             __global int *spikes)
{
    long gid = get_global_id(0);
    // Start index of neuron
    long ln_idx = gid * LN_LEN;
    // Index to store Vm and spike
    long out_idx = gid * input_len;
    
    for (int step = 0; step < input_len; step++) {
        lifNeurons[ln_idx + LN_I_INJECT_IDX] = Iinjects[step];
        
        if (lifNeurons[ln_idx + LN_N_STEPS_IN_REFR_IDX] > 0) {
            lifNeurons[ln_idx + LN_N_STEPS_IN_REFR_IDX] -= 1;
            lifNeurons[ln_idx + LN_HAS_FIRED_IDX] = 0;
        } else if (lifNeurons[ln_idx + LN_VM_IDX] >= lifNeurons[ln_idx + LN_V_THRESH_IDX]) {
            lifNeurons[ln_idx + LN_HAS_FIRED_IDX] = 1;
            lifNeurons[ln_idx + LN_N_STEPS_IN_REFR_IDX] =
                (float)ceil(lifNeurons[ln_idx + LN_T_REFRACT_IDX] / dt);
            lifNeurons[ln_idx + LN_VM_IDX] = lifNeurons[ln_idx + LN_V_RESET_IDX];
        } else {
            lifNeurons[ln_idx + LN_HAS_FIRED_IDX] = 0;
            lifNeurons[ln_idx + LN_SUMMATION_POINT_IDX] =
                lifNeurons[ln_idx + LN_SUMMATION_POINT_IDX] +
                lifNeurons[ln_idx + LN_I_INJECT_IDX] +
                (lifNeurons[ln_idx + LN_V_RESTING_IDX] / lifNeurons[ln_idx + LN_RM_IDX]);
            lifNeurons[ln_idx + LN_VM_IDX] =
                (lifNeurons[ln_idx + LN_C1_IDX] * lifNeurons[ln_idx + LN_VM_IDX]) +
                (lifNeurons[ln_idx + LN_C2_IDX] * lifNeurons[ln_idx + LN_SUMMATION_POINT_IDX]);
        }
        
        lifNeurons[ln_idx + LN_SUMMATION_POINT_IDX] = 0;
        
        Vms[out_idx + step] = lifNeurons[ln_idx + LN_VM_IDX];
        if (lifNeurons[ln_idx + LN_HAS_FIRED_IDX]) {
            spikes[out_idx + step] = 1;
        } else {
            spikes[out_idx + step] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
}

__kernel void nextState_plcr_pa(const float dt,
                                const int step,
                                const int input_len,
                                __global const float *Iinjects,
                                __global const float *lif_Rm,
                                __global const float *lif_Vresting,
                                __global const float *lif_Vthresh,
                                __global const float *lif_Vreset,
                                __global const float *lif_Vinit,
                                __global float *lif_Vm,
                                __global const float *lif_Trefract,
                                __global float *lif_summationPoint,
                                __global float *lif_Iinject,
                                __global const float *lif_C1,
                                __global const float *lif_C2,
                                __global int *lif_nStepsInRefr,
                                __global int *lif_hasFired,
                                __global float *Vms,
                                __global int *spikes)
{
    long gid = get_global_id(0);
    // Index to store Vm and spike
    long out_idx = gid * input_len;
    
    float Iinject = Iinjects[step];
    lif_Iinject[gid] = Iinject;
    int hasFired;
    float Vm = lif_Vm[gid];
    
    if (lif_nStepsInRefr[gid] > 0) {
        lif_nStepsInRefr[gid] -= 1;
        hasFired = 0;
    } else if (Vm >= lif_Vthresh[gid]) {
        hasFired = 1;
        lif_nStepsInRefr[gid] = ceil(lif_Trefract[gid] / dt);
        Vm = lif_Vreset[gid];
    } else {
        hasFired = 0;
        float summationPoint = lif_summationPoint[gid] + Iinject +
                               (lif_Vresting[gid] / lif_Rm[gid]);
        Vm = (lif_C1[gid] * Vm) + (lif_C2[gid] * summationPoint);
    }
    lif_hasFired[gid] = hasFired;
    lif_summationPoint[gid] = 0;
    
    lif_Vm[gid] = Vm;
    Vms[out_idx + step] = Vm;
    if (hasFired) {
        spikes[out_idx + step] = 1;
    } else {
        spikes[out_idx + step] = 0;
    }
}

__kernel void nextState_clcr_pa(const float dt,
                                const int input_len,
                                __global const float *Iinjects,
                                __global const float *lif_Rm,
                                __global const float *lif_Vresting,
                                __global const float *lif_Vthresh,
                                __global const float *lif_Vreset,
                                __global const float *lif_Vinit,
                                __global float *lif_Vm,
                                __global const float *lif_Trefract,
                                __global float *lif_summationPoint,
                                __global float *lif_Iinject,
                                __global const float *lif_C1,
                                __global const float *lif_C2,
                                __global int *lif_nStepsInRefr,
                                __global int *lif_hasFired,
                                __global float *Vms,
                                __global int *spikes)
{
    long gid = get_global_id(0);
    // Index to store Vm and spike
    long out_idx = gid * input_len;
    
    int nStepsInRefr = lif_nStepsInRefr[gid];
    int hasFired;
    float Vm = lif_Vm[gid];
    float summationPoint = lif_summationPoint[gid];
    
    for (int step = 0; step < input_len; step++) {
        if (nStepsInRefr > 0) {
            nStepsInRefr -= 1;
            hasFired = 0;
        } else if (Vm >= lif_Vthresh[gid]) {
            hasFired = 1;
            spikes[out_idx + step] = 1;
            nStepsInRefr = ceil(lif_Trefract[gid] / dt);
            Vm = lif_Vreset[gid];
        } else {
            hasFired = 0;
            summationPoint = summationPoint + Iinjects[step] +
                             (lif_Vresting[gid] / lif_Rm[gid]);
            Vm = (lif_C1[gid] * Vm) + (lif_C2[gid] * summationPoint);
        }
        summationPoint = 0;
        Vms[out_idx + step] = Vm;
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
}

#define BLOCK_START(pid, np, n) ((pid * n) / np)
#define BLOCK_END(pid, np, n) ((((pid + 1) * n) / np) - 1)

__kernel void nextState_clcrwg_pa(const float dt,
                                  const int input_len,
                                  __global const float *Iinjects,
                                  __global const float *lif_Rm,
                                  __global const float *lif_Vresting,
                                  __global const float *lif_Vthresh,
                                  __global const float *lif_Vreset,
                                  __global const float *lif_Vinit,
                                  __global float *lif_Vm,
                                  __global const float *lif_Trefract,
                                  __global float *lif_summationPoint,
                                  __global float *lif_Iinject,
                                  __global const float *lif_C1,
                                  __global const float *lif_C2,
                                  __global int *lif_nStepsInRefr,
                                  __global int *lif_hasFired,
                                  __global float *Vms,
                                  __global int *spikes)
{
    long grid = get_group_id(0);
    if (grid != 0) return;
    
    long gsz = get_global_size(0);
    long lsz = get_local_size(0);
    long lid = get_local_id(0);
    long start_idx = BLOCK_START(lid, lsz, gsz);
    long end_idx = BLOCK_END(lid, lsz, gsz);
    
    int input_len_lm = input_len;
    for (int step = 0; step < input_len_lm; step++) {
        for (long idx = start_idx; idx <= end_idx; idx++) {
            // Index to store Vm and spike
            long out_idx = idx * input_len_lm;
            
            int nStepsInRefr = lif_nStepsInRefr[idx];
            int hasFired;
            float Vm = lif_Vm[idx];
            float summationPoint = lif_summationPoint[idx];
            
            if (nStepsInRefr > 0) {
                nStepsInRefr -= 1;
                hasFired = 0;
            } else if (Vm >= lif_Vthresh[idx]) {
                hasFired = 1;
                spikes[out_idx + step] = 1;
                nStepsInRefr = ceil(lif_Trefract[idx] / dt);
                Vm = lif_Vreset[idx];
            } else {
                hasFired = 0;
                summationPoint = summationPoint + Iinjects[step] +
                                 (lif_Vresting[idx] / lif_Rm[idx]);
                Vm = (lif_C1[idx] * Vm) + (lif_C2[idx] * summationPoint);
            }
            Vms[out_idx + step] = Vm;
            lif_Vm[idx] = Vm;
            lif_nStepsInRefr[idx] = nStepsInRefr;
            lif_hasFired[idx] = hasFired;
            lif_summationPoint[idx] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
}
