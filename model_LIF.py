from spike_methods import *
from scipy import sparse
import copy
import math

class LIF_network():

    def __init__(self, state_initial, times_array, N, R, C, threshold,
                 last_firing_times, V_reset, refractory_time, g_syn_max,
                 E_syn, tau_syn, use_STDP, STDP_scaling, tau_W,
                 synapse_delay_delta_t, number_of_stored_most_recent_spikes,
                 number_of_stored_recent_per_neuron_presyn_network_spikes, bogus_spike_time,
                 spike_time_learning, memory_threshold,
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


        # ============ Stuff for neurons memorizing spike times of neighbors ==============================
        self.spike_time_learning = spike_time_learning
        self.number_of_stored_recent_per_neuron_presyn_network_spikes = number_of_stored_recent_per_neuron_presyn_network_spikes
        # same as recent_firing_times_list, except an array containing a set number of elements
        self.recent_firing_times_array = bogus_spike_time*np.ones((self.N,
                                                                   self.number_of_stored_recent_per_neuron_presyn_network_spikes)
                                                                   )

        self.number_of_presyn_neighbors = [{} for i in range(N)] # dict contains presyn_networks with value # of neighbors
        # list of self.N lists of
        # dicts of {network_presyn_name : {neighbor index within postsyn network of presyn i : list with
        # (# of postsynaptic spikes) number of ndarray of
        # most recent number_of_stored_recent_per_neuron_presyn_network_spikes firings}}
        self.neighbor_neuron_spike_time_list = [{} for i in range(self.N)]

        # dict of {presyn_network_name : list of postsyn neurons[list of presyn neuron indices within their own network]}
        self.neighbor_neuron_dict = {}

        self.memory_threshold = memory_threshold

        self.given_memories = False

        # ============ End of stuff for neurons memorizing spike times of neighbors ==============================


    def integrate_voltages_forward_by_dt(self, t, time_index, dt):
        self.alpha = dt / self.C
        self.beta = dt/(self.C*self.R)
        self.timesteps_synapse_delay = int(self.synapse_delay_delta_t / dt)

        def g_syn(g_syn_max, last_firing_times, tau_syn, t, synapse_delay_delta_t, recent_firing_times_list, number_of_stored_most_recent_spikes):

            N = len(recent_firing_times_list)
            recent_firing_times_array = self.bogus_spike_time*np.ones((N,number_of_stored_most_recent_spikes))
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
        # print("temp_1="+str(temp_1))
        # print("temp_2="+str(temp_2))
        # print("temp_3="+str(temp_3))
        # print("temp_4="+str(temp_4))
        # print("CURRENT I: "+str(I_ext(self.N, t)))

        #  ====================== Individual neuron spike memory section 1 ==============================

        if self.given_memories:
            print("Running given_memories section for t="+str(t))
            for n in range(self.N):  # For all neurons in this subnetwork
                sum = 0
                sigma_memory = 1  # TODO: Move this somewhere else

                # For presyn neurons within SAME subnetwork as postsyn
                presyn_network_name = self.name
                presyn_network = self
                number_of_spike_memories_stored = len(self.given_neighbor_neuron_spike_time_list[n][presyn_network_name])
                if number_of_spike_memories_stored != 0:
                    for i in range(self.number_of_presyn_neighbors[n][
                                       presyn_network_name]):  # for each neuron presynaptic to neuron n
                        presyn_network_neuron_index = self.neighbor_neuron_dict[presyn_network_name][n][i]  # get neuron n's label for presyn neuron
                        if self.W_sparse.tolil()[n, presyn_network_neuron_index] != 0:
                            for spike_memory_index in range(number_of_spike_memories_stored):
                                first_term_temp = (t-presyn_network.last_firing_times[presyn_network_neuron_index])
                                second_term_temp = self.given_neighbor_neuron_spike_time_list[n][presyn_network_name][
                                                       spike_memory_index][presyn_network_neuron_index]
                                mask_thresh = np.abs(self.bogus_spike_time) -5
                                # print(mask_thresh)
                                first_term = np.where(first_term_temp > mask_thresh, math.nan, first_term_temp) # make > mask_thresh into NaN
                                second_term = np.where(second_term_temp > mask_thresh, math.nan, second_term_temp) # make > mask_thresh into NaN
                                exp_power = np.exp(-np.fabs((first_term- second_term)) ** 2 / sigma_memory)
                                sum += np.sum(np.ma.array(exp_power,mask = np.isnan(exp_power))) # sum, ignoring NaNs
                                if t>18.8 and t<19.2:
                                    if sum>0.001:
                                        print("exp_power")
                                        print(exp_power)
                                        print("first_term_temp")
                                        print(first_term_temp)
                                        print("second_term_temp")
                                        print(second_term_temp)
                                        print("first_term")
                                        print(first_term)
                                        print("second_term")
                                        print(second_term)
                                        print("sum")
                                        print(sum)


                # For presyn neurons OUTSIDE subnetwork of postsyn
                for a_connection in self.dict_of_connected_networks.items(): # for all internetwork connections
                    # print("Check here")
                    network_x_of_connection = a_connection[1][0]
                    network_y_of_connection = a_connection[1][1]
                    is_self_network_x = (network_x_of_connection.name == self.name)
                    # default values set assuming self network is network y, so other network is network x
                    presyn_network_name = network_x_of_connection.name
                    number_of_spike_memories_stored = len(self.given_neighbor_neuron_spike_time_list[n][presyn_network_name])
                    if number_of_spike_memories_stored != 0:
                        presyn_network = network_x_of_connection
                        W_internetwork_sparse = internetwork_synapse_W.W_sparse_x_to_y
                        if is_self_network_x:
                            presyn_network_name = network_y_of_connection.name
                            presyn_network = network_y_of_connection
                            W_internetwork_sparse = internetwork_synapse_W.W_sparse_y_to_x
                        for i in range(self.number_of_presyn_neighbors[n][presyn_network_name]): # look at all presynaptic neurons
                            presyn_network_neuron_index = self.neighbor_neuron_dict[presyn_network_name][n][i]
                            if W_internetwork_sparse.tolil()[n, presyn_network_neuron_index] != 0: # if the presynaptic neuron does have a nonzero effect on the postsynaptic neuron n,
                                # print("HI!")
                                # print(presyn_network_neuron_index)
                                # print("HI2")
                                # print(n)
                                for spike_memory_index in range(number_of_spike_memories_stored):
                                    mask_thresh = 100#np.fabs(self.bogus_spike_time) - 5
                                    first_term = (t-presyn_network.last_firing_times[presyn_network_neuron_index])
                                    second_term = self.given_neighbor_neuron_spike_time_list[n][presyn_network_name][
                                                       spike_memory_index][
                                                       presyn_network_neuron_index]
                                    first_term = np.where(first_term > mask_thresh, math.nan, first_term)  # make > 999 into NaN
                                    print(first_term)
                                    second_term = np.where(second_term > mask_thresh, math.nan,
                                                           second_term)  # make > 999 into NaN
                                    exp_power = np.exp(-np.fabs((first_term - second_term)) ** 2 / sigma_memory)
                                    sum += np.sum(np.ma.array(exp_power, mask=np.isnan(exp_power)))
                # if sum>0:
                #     print("CHECKSUM: "+str(sum))
                if sum>1:
                    print("sum is "+str(sum))
                if sum > self.memory_threshold:  # memory sufficiently matches current observations
                    self.V_t[time_index + 1, n] = self.threshold + 1  # set voltage so that current neuron may spike
                    # print("Allowing to spike.")

        # ================== End of individual neuron spike memory section ===========================


        # Reset all neurons now (or still) in refractory to baseline
        self.in_refractory = np.multiply( ((t - self.last_firing_times) < self.refractory_time), ((t - self.last_firing_times) > 0.00001))
        (self.V_t[time_index+1,:])[self.in_refractory] = self.V_reset

        # Find V above threshold (find newly-spiking neurons)
        self.V_t_above_thresh = (self.V_t[time_index+1, :] > self.threshold)
        (self.V_t[time_index + 1, :])[self.V_t_above_thresh] = self.V_reset
        # print("ANY TRUE? "+str(np.any(self.V_t_above_thresh)))

        # Record firings
        self.last_firing_times[self.V_t_above_thresh] = t
        self.last_firing_array[time_index+1, :] = self.last_firing_times
        # if new last_firing_time (at time_index+1) isn't equal to that at time_index, then add last_firing_time to list
        for n in range(self.N):
            if self.last_firing_times[n] != self.last_firing_array[time_index, n]:
                self.recent_firing_times_list[n].append(self.last_firing_times[n])

        #  ====================== Individual neuron spike memory section 2 ==============================
        # Update data structures for neurons recording neighbor spike times
        if self.spike_time_learning == True:
            for n in range(self.N): # for every neuron in this network
                for m in range(len(self.recent_firing_times_list[n])): # for all most recent firing times of neuron n
                    # as long as this mth element is less than the length of recent_firing_times_array, which is
                    # length number_of_stored_recent_per_neuron_presyn_network_spikes:
                    if m < self.number_of_stored_recent_per_neuron_presyn_network_spikes:
                        # store firing time in array, with most recent on right and oldest on left:
                        self.recent_firing_times_array[n, -(m+1)] = self.recent_firing_times_list[n][-(m+1)]
                # If neuron has spiked, then update this neuron's memory of all other neurons' spike times
                if self.V_t_above_thresh[n] == True:
                    self.update_neighbor_spike_time_data_structure(n,t)

                # TODO: Increase neuron potential above threshold if presynaptic pattern which caused it to fire in past is present now

        #  ================== End of individual neuron spike memory section ===========================

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

    def initialize_neighbor_spike_time_data_structure(self):
        """Prepare data structure which will store each neuron's memory of its presynaptic neighbors' spike times.
            Actual values will be added during the forward integration of the full network
        """
        self.neighbor_neuron_spike_time_list = [{} for i in range(self.N)]
        for n in range(self.N):
            # For presyn neurons within SAME subnetwork as postsyn
            presyn_name = self.name
            self.neighbor_neuron_dict[presyn_name] = self.W_sparse.tolil().rows
            self.number_of_presyn_neighbors[n][presyn_name] = (self.W[n] != 0).sum()
            self.neighbor_neuron_spike_time_list[n][presyn_name] = []

            # For presyn neurons OUTSIDE subnetwork of postsyn
            for a_connection in self.dict_of_connected_networks.items():
                # print("Check here")
                network_x_of_connection = a_connection[1][0]
                network_y_of_connection = a_connection[1][1]
                internetwork_synapse_W = a_connection[1][2]
                is_self_network_x = (network_x_of_connection.name == self.name)
                presyn_name = network_x_of_connection.name
                W_internetwork = internetwork_synapse_W.W_x_to_y
                W_internetwork_sparse = internetwork_synapse_W.W_sparse_x_to_y
                if is_self_network_x:
                    W_internetwork = internetwork_synapse_W.W_y_to_x
                    W_internetwork_sparse = internetwork_synapse_W.W_sparse_y_to_x
                    presyn_name = network_y_of_connection.name
                self.neighbor_neuron_dict[presyn_name] = W_internetwork_sparse.tolil().rows # WARNING: Will count weights of 0 as connections
                # print(self.neighbor_neuron_dict[use_name])
                self.number_of_presyn_neighbors[n][presyn_name] = (W_internetwork[n] != 0).sum() # needed only for next line

                # Checking to make sure the chosen neighbor-counting method is correct (result: it works correctly):
                # if (len(self.neighbor_neuron_dict[use_name][n]) != self.number_of_presyn_neighbors[n][use_name]):
                #     print("No match!")
                #     print("Networks: "+str(network_x_of_connection.name)+" and "+str(network_y_of_connection.name))
                #     print("For n=" + str(n))
                #     print(W_internetwork_sparse)
                #     print(W_internetwork_sparse.toarray().sum())
                #     print("1st says n="+str(n) +" is connected to "+str(len(self.neighbor_neuron_dict[use_name][n]))+" presyn;"+" 2nd says "+str((W_internetwork[n] != 0).sum()))
                #     print("1st: "+str(self.neighbor_neuron_dict[use_name][n]))
                #     print("2nd: "+str(W_internetwork))
                #     print(W_internetwork[n])
                #     print("\n")

                self.neighbor_neuron_spike_time_list[n][presyn_name] = []


    def update_neighbor_spike_time_data_structure(self, n, t):
        """
        Updates self.neighbor_neuron_spike_time_list
        Value n is an integer index of neuron in self network that has just spiked
        """
        # For presyn neurons within SAME subnetwork as postsyn\
        presyn_network_name = self.name
        self.neighbor_neuron_spike_time_list[n][presyn_network_name].append({})
        for i in range(self.number_of_presyn_neighbors[n][presyn_network_name]): # for each neuron presynaptic to neuron n
            presyn_network_neuron_index = self.neighbor_neuron_dict[presyn_network_name][n][i] # get neuron n's label for presyn neuron
            self.neighbor_neuron_spike_time_list[n][presyn_network_name][-1][presyn_network_neuron_index] = t-copy.deepcopy(self.recent_firing_times_array[presyn_network_neuron_index, -self.number_of_stored_recent_per_neuron_presyn_network_spikes:])

        # For presyn neurons OUTSIDE subnetwork of postsyn
        for a_connection in self.dict_of_connected_networks.items():
            # print("Check here")
            network_x_of_connection = a_connection[1][0]
            network_y_of_connection = a_connection[1][1]
            is_self_network_x = (network_x_of_connection.name == self.name)
            # default values set assuming self network is network y, so other network is network x
            presyn_network_name = network_x_of_connection.name
            presyn_network = network_x_of_connection
            if is_self_network_x:
                presyn_network_name = network_y_of_connection.name
                presyn_network = network_y_of_connection
            self.neighbor_neuron_spike_time_list[n][presyn_network_name].append({})
            for i in range(self.number_of_presyn_neighbors[n][presyn_network_name]):
                presyn_network_neuron_index = self.neighbor_neuron_dict[presyn_network_name][n][i]
                self.neighbor_neuron_spike_time_list[n][presyn_network_name][-1][presyn_network_neuron_index] = t-copy.deepcopy(presyn_network.recent_firing_times_array[
                                                                                     presyn_network_neuron_index,
                                                                                     -self.number_of_stored_recent_per_neuron_presyn_network_spikes:])

    def return_results(self):
        self.spike_list = process_spike_results(self.last_firing_array, self.bogus_spike_time, self.N)

        # just for saving stuff:
        # spike_list_array = spike_list_to_array(self.N, spike_list)  # array dimension(N, ???)
        # np.savetxt('spike_data/spike_list;' + extra_descriptors + '.txt', spike_list_array, fmt='%.3e')

        return [self.V_t, self.W, self.times_array, self.spike_list]

    def copy_neuron_memories(self, taught_network):
        self.given_memories = True
        self.spike_time_learning = True
        self.given_neighbor_neuron_spike_time_list = taught_network.neighbor_neuron_spike_time_list
        self.spike_time_learning = taught_network.spike_time_learning
        self.number_of_stored_recent_per_neuron_presyn_network_spikes = taught_network.number_of_stored_recent_per_neuron_presyn_network_spikes
        # same as recent_firing_times_list, except an array containing a set number of elements
        self.recent_firing_times_array = taught_network.recent_firing_times_array

        self.number_of_presyn_neighbors = taught_network.number_of_presyn_neighbors
        # list of self.N lists of
        # dicts of {network_presyn_name : {neighbor index within postsyn network of presyn i : list with
        # (# of postsynaptic spikes) number of ndarray of
        # most recent number_of_stored_recent_per_neuron_presyn_network_spikes firings}}
        self.neighbor_neuron_spike_time_list = taught_network.neighbor_neuron_spike_time_list

        # dict of {presyn_network_name : list of postsyn neurons[list of presyn neuron indices within their own network]}
        self.neighbor_neuron_dict = taught_network.neighbor_neuron_dict

        self.memory_threshold = taught_network.memory_threshold