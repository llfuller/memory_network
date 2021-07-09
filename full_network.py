import numpy as np

# A full network constructed by linking together smaller networks

class full_network():

    def __init__(self):
        self.dict_of_networks = {}
        self.list_of_internetwork_synapses = []
        self.list_of_connected_network_pairs = []

    def set_subnetworks(self, dict_of_networks):
        self.dict_of_networks.update(dict_of_networks)

    def connect_to_input_data(self, data_function, network_y):
        # connect networks to stimulus introduced by artificial wires or
        # from sensory organs such as the ear.
        # data_function should take time t as an argument
        network_y.list_of_external_stimulus_functions.append(data_function)

    def connect_two_networks(self, network_x_name, network_y_name, STDP_method = None):
        # Store in full_network member list all links between subnetworks for later
        # updating if STDP != None
        internetwork_synapse_W = None # put a synapse W generation function here later
        self.list_of_internetwork_synapses.append([network_x_name, network_y_name,
                                                   internetwork_synapse_W, STDP_method])
        # Store references to network y in network x (and vice versa) for easy access during
        # voltage updates
        self.dict_of_networks['network_x_name'].\
            list_of_connected_networks.append(self.dict_of_networks['network_y_name'])
        self.dict_of_networks['network_y_name']. \
            list_of_connected_networks.append(self.dict_of_networks['network_x_name'])

    def run(self, times_array, dt):
        for time_index, t in enumerate(self.times_array[:-1]):
            # progress full network voltage a single timestep
            for a_subnetwork in self.list_of_networks:
                a_subnetwork.integrate_voltages_forward_by_dt(t, time_index, dt)
            # progress subnetwork internal synaptic connections a single timestep
            for a_subnetwork in self.list_of_networks:
                a_subnetwork.integrate_internal_W_forward_by_dt(t, time_index, dt)
            # progress network-to-network synaptic connections in a single timestep
            for a_set_of_connections in self.list_of_internetwork_synapses:
                a_set_of_connections.integrate_external_W_forward_by_dt(t, time_index, dt)

    def organize_results(self):
        result_list = []
        for a_subnetwork in self.dict_of_networks.values():
            result_list.append(a_subnetwork.return_results())






