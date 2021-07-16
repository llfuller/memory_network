from spike_methods import *
from scipy import sparse


class LIF_network():

    def __init__(self, state_initial, times_array, N, R, C, threshold,
                 last_firing_times, V_reset, refractory_time, g_syn_max,
                 E_syn, tau_syn, use_STDP, STDP_scaling, tau_W,
                 synapse_delay_delta_t, number_of_stored_most_recent_spikes, bogus_spike_time,
                 network_name):
        self.state_initial = state_initial
        self.times_array = times_array
        self.N = N
        self.R = R
        self.C = C
        self.threshold = threshold
        self.last_firing_times = last_firing_times
        self.V_reset = V_reset
        self.refractory_time = refractory_time
        self.g_syn_max = g_syn_max
        self.E_syn = E_syn
        self.tau_syn = tau_syn
        self.use_STDP = use_STDP
        self.STDP_scaling = STDP_scaling
        self.tau_W = tau_W
        self.synapse_delay_delta_t = synapse_delay_delta_t
        self.V_t = np.zeros((self.times_array.shape[0], self.N))
        self.V_t[0, :] = self.state_initial[:self.N]
        self.W = self.state_initial[self.N : self.N+self.N*self.N].reshape((self.N,self.N)) # used for integrating W
        self.W_sparse = sparse.csr_matrix(self.W) # used for integrating voltages
        self.last_firing_array = np.nan*np.ones((self.times_array.shape[0], self.N)) # nan allows easier raster plotting later
        self.delta_W = None

        self.list_of_external_stimulus_functions = []
        # self.list_of_connected_networks = []
        self.dict_of_connected_networks = {} # 'network name' : [network, internetwork_connections]

        # alpha, beta, and synaptic timestep (not time) delay can only be determined after dt is submitted
        self.alpha = None
        self.beta = None
        self.timesteps_synapse_delay = None

        self.in_refractory = None
        self.V_t_above_thresh = None
        self.V_t_below_reset = None

        # bogus stand-in spike time to be replaced later in the computation
        self.bogus_spike_time = bogus_spike_time

        self.spike_list = [[] for i in range(N)]  # Filled in return_results(self).
        # Need N separate empty lists to record spike times to

        # Setting max number of recent past spikes to check when calculating synaptic conductance
        self.number_of_stored_most_recent_spikes = number_of_stored_most_recent_spikes
        self.recent_firing_times_list = [[] for i in range(N)]

        self.name = network_name


    def integrate_voltages_forward_by_dt(self, t, time_index, dt):
        self.alpha = dt / self.C
        self.beta = dt/(self.C*self.R)
        self.timesteps_synapse_delay = int(self.synapse_delay_delta_t / dt)

        def g_syn(g_syn_max, last_firing_times, tau_syn, t, synapse_delay_delta_t, recent_firing_times_list, number_of_stored_most_recent_spikes):

            N = len(recent_firing_times_list)
            recent_firing_times_array = -1000*np.ones((N,number_of_stored_most_recent_spikes))
            for n in range(N):
                recent_firing_times_list_row_n_array = np.array(recent_firing_times_list[n])
                length_of_list = len(recent_firing_times_list[n])
                for j in range(length_of_list):
                    if j<number_of_stored_most_recent_spikes:
                        # Get the most recent spikes
                        recent_firing_times_array[n,j] = recent_firing_times_list_row_n_array[-j]
            # print(recent_firing_times_array)
            # print(np.shape(recent_firing_times_array))
            t_after_firing = t - (recent_firing_times_array + synapse_delay_delta_t)
            # print(np.shape(t_after_firing))
            # print(np.shape(t_after_firing_2))
            synapse_effect_on = t_after_firing.copy()
            synapse_effect_on[t_after_firing<0] = 0
            synapse_effect_on[t_after_firing>0] = 1

            return np.sum(g_syn_max * np.multiply(np.exp(-t_after_firing / tau_syn),synapse_effect_on), axis=1)

        # Leaky integrate (decay + external lab currents; note external currents can create asynchronous firing)
        self.V_t[time_index+1,:] = (1-self.beta)*self.V_t[time_index,:]
        # External stimuli (from wire or sensing organ):
        for I_ext in self.list_of_external_stimulus_functions:
            self.V_t[time_index+1,:] += self.alpha * I_ext(self.N,t)
        # Stimuli from other networks
        for a_connection in self.dict_of_connected_networks.items():
            # print("Check here")
            network_x_of_connection = a_connection[1][0]
            network_y_of_connection = a_connection[1][1]
            # print(self.dict_of_connected_networks)
            # print(upstream_network_in_connection)

            # if names match, then self is the upstream connection
            is_self_network_x = (network_x_of_connection.name == self.name)
            # print(self.name + " is upstream: " + str(is_self_upstream))
            internetwork_synapse_W = a_connection[1][2]
            # default is case where self is downstream
            W_internetwork_sparse = internetwork_synapse_W.W_sparse_x_to_y
            E_syn_internetwork = network_x_of_connection.E_syn
            last_firing_time_other_network = network_x_of_connection.last_firing_times
            synapse_delay_delta_t_connection = internetwork_synapse_W.synapse_delay_delta_t_x_to_y
            recent_firing_times_list_internetwork = network_x_of_connection.recent_firing_times_list
            number_of_stored_most_recent_spikes_internetwork = network_x_of_connection.number_of_stored_most_recent_spikes

            if is_self_network_x:
                W_internetwork_sparse = internetwork_synapse_W.W_sparse_y_to_x
                E_syn_internetwork = network_y_of_connection.E_syn
                last_firing_time_other_network = network_y_of_connection.last_firing_times
                synapse_delay_delta_t_connection = internetwork_synapse_W.synapse_delay_delta_t_y_to_x
                recent_firing_times_list_internetwork = network_y_of_connection.recent_firing_times_list
                number_of_stored_most_recent_spikes_internetwork = network_y_of_connection.number_of_stored_most_recent_spikes

            synapse_delay_timesteps_connection = int(synapse_delay_delta_t_connection / dt)
            temp_1 = W_internetwork_sparse.multiply(
                g_syn(internetwork_synapse_W.g_syn_max, last_firing_time_other_network, internetwork_synapse_W.tau_syn,
                      t, synapse_delay_delta_t_connection, recent_firing_times_list_internetwork, number_of_stored_most_recent_spikes_internetwork))

            # E_syn still needs to be figured out
            temp_2 = temp_1.multiply(
                np.subtract.outer(E_syn_internetwork, self.V_t[time_index - synapse_delay_timesteps_connection]).transpose())
            temp_3 = np.sum(temp_2, axis=1)
            temp_4 = np.squeeze(np.array(temp_3))  # need to set as array for this to work
            self.V_t[time_index + 1, :] += self.alpha * temp_4

        # Add synaptic currents due to other neurons firing
        temp_1 = self.W_sparse.multiply(g_syn(self.g_syn_max, self.last_firing_times, self.tau_syn,
                                              t, self.synapse_delay_delta_t, self.recent_firing_times_list,
                                              self.number_of_stored_most_recent_spikes))
        temp_2 = temp_1.multiply(np.subtract.outer(self.E_syn,self.V_t[time_index-self.timesteps_synapse_delay]).transpose())
        temp_3 = np.sum(temp_2, axis=1)
        temp_4 = np.squeeze(np.array(temp_3)) # need to set as array for this to work
        self.V_t[time_index + 1, :] += self.alpha * temp_4

        # Reset all neurons now (or still) in refractory to baseline
        self.in_refractory = np.multiply( ((t - self.last_firing_times) < self.refractory_time), ((t - self.last_firing_times) > 0.00001))
        (self.V_t[time_index+1,:])[self.in_refractory] = self.V_reset

        # Find V above threshold (find newly-spiking neurons)
        self.V_t_above_thresh = (self.V_t[time_index+1, :] > self.threshold)
        (self.V_t[time_index + 1, :])[self.V_t_above_thresh] = self.V_reset

        # Record firings
        self.last_firing_times[self.V_t_above_thresh] = t
        self.last_firing_array[time_index+1, :] = self.last_firing_times
        # if new last_firing_time (at time_index+1) isn't equal to that at time_index, then add last_firing_time to list
        for n in range(self.N):
            if self.last_firing_times[n] != self.last_firing_array[time_index, n]:
                self.recent_firing_times_list[n].append(self.last_firing_times[n])

        # Reset to baseline if below baseline:
        self.V_t_below_reset = (self.V_t[time_index+1, :] <= self.V_reset)
        self.V_t[time_index + 1, :][self.V_t_below_reset] = self.V_reset

    def integrate_internal_W_forward_by_dt(self, t, time_index, dt):
        # Adapt weights into postsynaptic neurons (index i) that have just fired according to V_t_above_thresh
        if self.use_STDP:
            self.timing_difference = np.subtract.outer(self.last_firing_times, self.last_firing_times)  # positive if pre spiked first
            # self.timing_difference[self.timing_difference<0] = 0
            self.delta_W = self.STDP_scaling \
                           * np.multiply(np.sign(self.timing_difference), np.exp(-np.abs(self.timing_difference)/self.tau_W))
            self.delta_W[self.W==0] = 0 # Make sure cells without connections do not develop them
            self.delta_W[self.delta_W<0] = 0 # Ensure conductances don't drop below zero (which would switch excite/inhibit)
            self.W += self.delta_W
            self.W_sparse = sparse.csr_matrix(self.W)

    def return_results(self):
        self.spike_list = process_spike_results(self.last_firing_array, self.bogus_spike_time, self.N)

        # just for saving stuff:
        # spike_list_array = spike_list_to_array(self.N, spike_list)  # array dimension(N, ???)
        # np.savetxt('spike_data/spike_list;' + extra_descriptors + '.txt', spike_list_array, fmt='%.3e')

        return [self.V_t, self.W, self.times_array, self.spike_list]