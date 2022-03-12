import numpy as np
import matplotlib.pyplot as plt

class LIF_network():

    def __init__(self, state_initial, times_array, N, I_instr_t, R, C, threshold,
                 last_firing_times, V_reset, refractory_time, g_syn_max,
                 E_syn, tau_syn, spike_list, use_STDP, STDP_scaling, tau_W, synapse_delay_delta_t):
        self.state_initial = state_initial
        self.times_array = times_array
        self.N = N
        self.I_instr_t = I_instr_t
        self.R = R
        self.C = C
        self.threshold = threshold
        self.last_firing_times = last_firing_times
        self.V_reset = V_reset
        self.refractory_time = refractory_time
        self.g_syn_max = g_syn_max
        self.E_syn = E_syn
        self.tau_syn = tau_syn
        self.spike_list = spike_list
        self.use_STDP = use_STDP
        self.STDP_scaling = STDP_scaling
        self.tau_W = tau_W
        self.synapse_delay_delta_t = synapse_delay_delta_t
        self.V_t = None
        self.W = None
        self.last_firing_array = None
        self.timesteps_synapse_delay = None
        self.delta_W = None

    def integrate_forward_LIF_network(self, dt):
        alpha = dt / self.C
        beta = dt/(self.C*self.R)
        self.V_t = np.zeros((self.times_array.shape[0], self.N))
        self.V_t[0,:] = self.state_initial[:self.N]
        self.last_firing_array = np.nan*np.ones((self.times_array.shape[0], self.N)) # nan allows easier raster plotting later
        self.W = self.state_initial[self.N : self.N+self.N*self.N].reshape((self.N,self.N))
        self.timesteps_synapse_delay = int(self.synapse_delay_delta_t/dt)
        def g_syn(g_syn_max, last_firing_times, tau_syn, t, synapse_delay_delta_t):
            t_after_firing = t - (last_firing_times + synapse_delay_delta_t)
            synapse_effect_on = t_after_firing.copy()
            synapse_effect_on[t_after_firing<0] = 0
            synapse_effect_on[t_after_firing>0] = 1

            return g_syn_max * np.multiply(np.exp(-t_after_firing / tau_syn),synapse_effect_on)

        for time_index, t in enumerate(self.times_array[:-1]):
            # Leaky integrate (decay + external lab currents; note external currents can create asynchronous firing)
            self.V_t[time_index+1,:] = (1-beta)*self.V_t[time_index,:]
            self.V_t[time_index+1,:] += alpha * self.I_instr_t(self.N,t)
            # Add synaptic currents due to firing
            self.V_t[time_index + 1, :] += alpha * \
                                      np.multiply(np.matmul(self.W,
                                                            g_syn(self.g_syn_max, self.last_firing_times, self.tau_syn, t,
                                                                  self.synapse_delay_delta_t)),
                                                  (self.E_syn-self.V_t[time_index-self.timesteps_synapse_delay]))

            # Reset all neurons now (or still) in refractory to baseline
            in_refractory = np.multiply( ((t - self.last_firing_times) < self.refractory_time), ((t - self.last_firing_times) > 0.00001))
            (self.V_t[time_index+1,:])[in_refractory] = self.V_reset

            # Find V above threshold (find newly-spiking neurons)
            V_t_above_thresh = (self.V_t[time_index+1, :] > self.threshold)
            (self.V_t[time_index + 1, :])[V_t_above_thresh] = self.V_reset

            # Record firings
            self.last_firing_times[V_t_above_thresh] = t
            self.last_firing_array[time_index+1, :] = self.last_firing_times

            # Reset to baseline if below baseline:
            V_t_below_reset = (self.V_t[time_index+1, :] <= self.V_reset)
            self.V_t[time_index + 1, :][V_t_below_reset] = self.V_reset

            # Adapt weights into postsynaptic neurons (index i) that have just fired according to V_t_above_thresh
            if self.use_STDP:
                timing_difference = np.subtract.outer(self.last_firing_times, self.last_firing_times)  # positive if pre spiked first
                # timing_difference[timing_difference<0] = 0
                self.delta_W = self.STDP_scaling \
                               * np.multiply(np.sign(timing_difference), np.exp(-np.abs(timing_difference)/self.tau_W))
                self.delta_W[self.W==0] = 0 # Make sure cells without connections do not develop them
                self.delta_W[self.delta_W<0] = 0 # Ensure conductances don't drop below zero (which would switch excite/inhibit)
                self.W += self.delta_W
        return [self.V_t, self.W, self.last_firing_array]