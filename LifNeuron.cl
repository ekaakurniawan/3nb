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


__kernel void nextState_pL_rI(__global const float *dt,
                              __global const float *Iinjects,
                              __global int *step,
                              __global float *lifNeurons,
                              __global float *Vms,
                              __global int *spikes)
{
    long gid = get_global_id(0);
    long ln_idx = gid * LN_LEN;
    
    float dt_loc = dt[0];
    int step_loc = step[0];
    lifNeurons[ln_idx + LN_I_INJECT_IDX] = Iinjects[step_loc];
    
    if (lifNeurons[ln_idx + LN_N_STEPS_IN_REFR_IDX] > 0) {
        lifNeurons[ln_idx + LN_N_STEPS_IN_REFR_IDX] -= 1;
        lifNeurons[ln_idx + LN_HAS_FIRED_IDX] = 0;
    } else if (lifNeurons[ln_idx + LN_VM_IDX] >= lifNeurons[ln_idx + LN_V_THRESH_IDX]) {
        lifNeurons[ln_idx + LN_HAS_FIRED_IDX] = 1;
        lifNeurons[ln_idx + LN_N_STEPS_IN_REFR_IDX] =
            (float)ceil(lifNeurons[ln_idx + LN_T_REFRACT_IDX] / dt_loc);
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
    
    if (gid == 0) step[0] = step_loc + 1;
}

__kernel void nextState_pL_rC(__global const float *dt,
                              __global const float *Iinjects,
                              __global const int *input_len,
                              __global int *step,
                              __global float *lifNeurons,
                              __global float *Vms,
                              __global int *spikes)
{
    long gid = get_global_id(0);
    // Start index of neuron
    long ln_idx = gid * LN_LEN;
    // Index to store Vm and spike
    long out_idx = gid * input_len[0];
    
    float dt_loc = dt[0];
    int step_loc = step[0];
    lifNeurons[ln_idx + LN_I_INJECT_IDX] = Iinjects[step_loc];
    
    if (lifNeurons[ln_idx + LN_N_STEPS_IN_REFR_IDX] > 0) {
        lifNeurons[ln_idx + LN_N_STEPS_IN_REFR_IDX] -= 1;
        lifNeurons[ln_idx + LN_HAS_FIRED_IDX] = 0;
    } else if (lifNeurons[ln_idx + LN_VM_IDX] >= lifNeurons[ln_idx + LN_V_THRESH_IDX]) {
        lifNeurons[ln_idx + LN_HAS_FIRED_IDX] = 1;
        lifNeurons[ln_idx + LN_N_STEPS_IN_REFR_IDX] =
            (float)ceil(lifNeurons[ln_idx + LN_T_REFRACT_IDX] / dt_loc);
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
    
    Vms[out_idx + step_loc] = lifNeurons[ln_idx + LN_VM_IDX];
    if (lifNeurons[ln_idx + LN_HAS_FIRED_IDX]) {
        spikes[out_idx + step_loc] = 1;
    } else {
        spikes[out_idx + step_loc] = 0;
    }
}

__kernel void nextState_cL(__global const float *dt,
                           __global const float *Iinjects,
                           __global const int *input_len,
                           __global float *lifNeurons,
                           __global float *Vms,
                           __global int *spikes)
{
    long gid = get_global_id(0);
    // Start index of neuron
    long ln_idx = gid * LN_LEN;
    // Index to store Vm and spike
    int input_len_loc = input_len[0];
    long out_idx = gid * input_len_loc;
    
    float dt_loc = dt[0];
    
    for (int step = 0; step < input_len_loc; step++) {
        lifNeurons[ln_idx + LN_I_INJECT_IDX] = Iinjects[step];
        
        if (lifNeurons[ln_idx + LN_N_STEPS_IN_REFR_IDX] > 0) {
            lifNeurons[ln_idx + LN_N_STEPS_IN_REFR_IDX] -= 1;
            lifNeurons[ln_idx + LN_HAS_FIRED_IDX] = 0;
        } else if (lifNeurons[ln_idx + LN_VM_IDX] >= lifNeurons[ln_idx + LN_V_THRESH_IDX]) {
            lifNeurons[ln_idx + LN_HAS_FIRED_IDX] = 1;
            lifNeurons[ln_idx + LN_N_STEPS_IN_REFR_IDX] =
                (float)ceil(lifNeurons[ln_idx + LN_T_REFRACT_IDX] / dt_loc);
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
}
