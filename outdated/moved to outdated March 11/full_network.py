import synaptic_weights
# A full network constructed by linking together smaller networks

class full_network():

    def __init__(self):
        self.dict_of_networks = {}
        self.list_of_internetwork_synapses = []
        self.list_of_connected_network_pairs = []

    def set_subnetworks(self, dict_of_networks):
        self.dict_of_networks.update(dict_of_networks)

    def connect_to_input_data(self, data_function, network_y):
        # Connect networks to stimulus introduced by artificial wires or
        # from sensory organs such as the ear.
        # data_function should take time t as an argument
        # Note: Networks don't need to be part of this full network to be linked to data by this method
        #       Perhaps that should be changed
        network_y.list_of_external_stimulus_functions.append(data_function)

    def connect_two_networks(self, network_x, network_y,
                             synapse_density, l_bound, stats_scale,
                             STDP_method = None, g_x_to_y_and_y_to_x = [1,0]):
        """
        :param network_x: (type network object; ex: LIF_network)
        :param network_y: (type network object; ex: LIF_network)
        :param synapse_density: (type: float) (1 = fully connected, 0 = never any connection)
        :param l_bound: (type: float) Synaptic weight bounds (dimensionless)
        :param stats_scale: (type: float) u_bound - l_bound
        :param STDP_method:
        """
        # Store in full_network member list all links between subnetworks for later
        # updating if STDP != None
        internetwork_synapse_W = synaptic_weights.internetwork_weights()
        internetwork_synapse_W.make_network_to_network_weights(
            network_x,
            network_y,
            synapse_density,
            l_bound,
            stats_scale,
            STDP_method,
            g_x_to_y_and_y_to_x = g_x_to_y_and_y_to_x
        ) # put a synapse W generation function here later

        new_connection = [network_x, network_y, internetwork_synapse_W, STDP_method]
        self.list_of_internetwork_synapses.append(new_connection)
        # Store references to network y in network x (and vice versa) for easy access during
        # voltage updates
        network_x.dict_of_connected_networks[network_y.name] = new_connection
        network_y.dict_of_connected_networks[network_x.name] = new_connection

    def run(self, times_array, dt):
        """Computes state of full network at all times in times_array

        :param times_array: array of all times network states are computed for; units are ms
        :param dt: size of timesteps in ms
        """
        # set up data structure allowing neurons to record presynaptic neighbor's previous firing times
        for a_subnetwork in self.dict_of_networks.values():
            a_subnetwork.initialize_neighbor_spike_time_data_structure()

        # integrate networks forward
        for time_index, t in enumerate(times_array[:-1]):
            # progress full network voltage a single timestep
            for a_subnetwork in self.dict_of_networks.values():
                a_subnetwork.integrate_voltages_forward_by_dt(t, time_index, dt)
            # progress subnetwork internal synaptic connections a single timestep
            for a_subnetwork in self.dict_of_networks.values():
                a_subnetwork.integrate_internal_W_forward_by_dt(t, time_index, dt)
            # progress network-to-network synaptic connections in a single timestep
            for a_list_for_connections in self.list_of_internetwork_synapses:
                internetwork_synapse_W = a_list_for_connections[2]
                internetwork_synapse_W.integrate_external_W_forward_by_dt(t, time_index, dt)

    def organize_results(self):
        """
        :returns all subnetwork voltages, spikes, etc in dict.
        Key is name of network.
        Value is a list of lists:
        List layer 0 is each subnetwork of the full_network.
        List layer 1 is voltages, spikes, etc.
        """
        result_dict = {}
        for a_subnetwork in self.dict_of_networks.values():
            result_dict[a_subnetwork.name] = a_subnetwork.return_results()
        return result_dict
