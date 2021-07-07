import numpy as np
import matplotlib.pyplot as plt

def PST_LIF_1_network(state_initial, times_array,
                dt, N, I_instr_t, R, C, threshold,
                last_firing_times, V_reset, refractory_time, g_syn_max,
                E_syn, tau_syn, spike_list, use_STDP, STDP_scaling, tau_W, synapse_delay_delta_t):
    alpha = dt / C
    beta = dt/(C*R)
    V_t = np.zeros((times_array.shape[0], N))
    V_t[0,:] = state_initial[:N]
    last_firing_array = np.nan*np.ones((times_array.shape[0], N)) # nan allows easier raster plotting later
    W = state_initial[N : N+N*N].reshape((N,N))
    timesteps_synapse_delay = int(synapse_delay_delta_t/dt)
    def g_syn(g_syn_max, last_firing_times, tau_syn, t, synapse_delay_delta_t):
        t_after_firing = t - (last_firing_times + synapse_delay_delta_t)
        synapse_effect_on = t_after_firing.copy()
        synapse_effect_on[t_after_firing<0] = 0
        synapse_effect_on[t_after_firing>0] = 1

        return g_syn_max * np.multiply(np.exp(-t_after_firing / tau_syn),synapse_effect_on)

    for time_index, t in enumerate(times_array[:-1]):
        # Leaky integrate (decay + external lab currents; note external currents can create asynchronous firing)
        V_t[time_index+1,:] = (1-beta)*V_t[time_index,:]
        V_t[time_index+1,:] += alpha * I_instr_t(N,t)
        # Add synaptic currents due to firing
        V_t[time_index + 1, :] += alpha * \
                                  np.multiply(np.matmul(W,
                                                        g_syn(g_syn_max, last_firing_times, tau_syn, t,
                                                              synapse_delay_delta_t)),
                                              (E_syn-V_t[time_index-timesteps_synapse_delay]))

        # Reset all neurons now (or still) in refractory to baseline
        in_refractory = np.multiply( ((t - last_firing_times) < refractory_time), ((t - last_firing_times) > 0.00001))
        (V_t[time_index+1,:])[in_refractory] = V_reset

        # Find V above threshold (find newly-spiking neurons)
        V_t_above_thresh = (V_t[time_index+1, :] > threshold)
        (V_t[time_index + 1, :])[V_t_above_thresh] = V_reset

        # Record firings
        last_firing_times[V_t_above_thresh] = t
        last_firing_array[time_index+1, :] = last_firing_times

        # Reset to baseline if below baseline:
        V_t_below_reset = (V_t[time_index+1, :] <= V_reset)
        V_t[time_index + 1, :][V_t_below_reset] = V_reset

        # Adapt weights into postsynaptic neurons (index i) that have just fired according to V_t_above_thresh
        if use_STDP:
            timing_difference = np.subtract.outer(last_firing_times, last_firing_times)  # positive if pre spiked first
            # timing_difference[timing_difference<0] = 0
            delta_W = STDP_scaling \
                           * np.multiply(np.sign(timing_difference), np.exp(-np.abs(timing_difference)/tau_W))
            delta_W[W==0] = 0 # Make sure cells without connections do not develop them
            delta_W[delta_W<0] = 0 # Ensure conductances don't drop below zero (which would switch excite/inhibit)
            W += delta_W
    return [V_t, W, last_firing_array]