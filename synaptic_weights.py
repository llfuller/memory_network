
import scipy
import numpy as np
def make_internal_weights(N, synapse_density, l_bound, stats_scale):
    """
    :param N: Number of neurons in network
    :param synapse_density: 1 = fully connected, 0 = never any connection
    :param l_bound: lower Synaptic weight bounds (dimensionless)
    :param stats_scale: used for "scale" argument in data_rvs argument of scipy sparse random method
    :return: uniformly randomized synaptic weights between neurons within a single network, with zeros on diagonal
    """
    state_initial_synaptic = scipy.sparse.random(N, N, density=synapse_density,
                                                 data_rvs=scipy.stats.uniform(loc=l_bound, scale=stats_scale).rvs)
    # remove diagonal synaptic weights
    matrix_initial_synapse_states = scipy.sparse.csr_matrix(state_initial_synaptic).toarray()
    np.fill_diagonal(matrix_initial_synapse_states, val=0)
    return matrix_initial_synapse_states

class internetwork_weights():
    def __init__(self):
        self.W_x_to_y = None
        self.W_sparse_x_to_y = None
        self.synapse_delay_delta_t_x_to_y = 7

        self.W_y_to_x = None
        self.W_sparse_y_to_x = None
        self.synapse_delay_delta_t_y_to_x = 7


        self.STDP_method = None
        self.use_STDP = None
        self.network_x = None
        self.network_y = None

        self.g_syn_max = 1
        self.tau_syn = 10


    def make_network_to_network_weights(self, network_x, network_y, synapse_density, l_bound, stats_scale, STDP_method):
        """
        :param network_x: (type network object; ex: LIF_network) Closer to sensory data ("upstream")
        :param network_y: (type network object; ex: LIF_network) Further from sensory data ("downstream")
        :param synapse_density: 1 = fully connected, 0 = never any connection
        :param l_bound: lower Synaptic weight bounds (dimensionless)
        :param stats_scale: used for "scale" argument in data_rvs argument of scipy sparse random method
        :param STDP_method: (type: boolean)
        :return: uniformly randomized synaptic weights between neurons between two networks, with zeros on diagonal
        """
        N_x = network_x.N
        N_y = network_y.N
        state_initial_synaptic = scipy.sparse.random(N_y, N_x, density=synapse_density,
                                                     data_rvs=scipy.stats.uniform(loc=l_bound, scale=stats_scale).rvs)
        self.W_sparse_x_to_y = scipy.sparse.csr_matrix(state_initial_synaptic)
        self.W_x_to_y = self.W_sparse_x_to_y.toarray()

        state_initial_synaptic = scipy.sparse.random(N_x, N_y, density=synapse_density,
                                                     data_rvs=scipy.stats.uniform(loc=l_bound, scale=stats_scale).rvs)
        self.W_sparse_y_to_x = scipy.sparse.csr_matrix(state_initial_synaptic)
        self.W_y_to_x = self.W_sparse_y_to_x.toarray()


        self.STDP_method = STDP_method
        if STDP_method == None or STDP_method == False:
            self.use_STDP = False


    def integrate_external_W_forward_by_dt(self, t, time_index, dt):
        # Adapt weights into postsynaptic neurons (index i) that have just fired according to V_t_above_thresh
        if self.use_STDP == False:
            pass
        else:
            pass
