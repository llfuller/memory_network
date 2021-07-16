from model_LIF import LIF_network
import numpy as np
import synaptic_weights


########################################################################################################################
#
# Runs N neurons and N*N synapses in parallel ( system with dimensions (2*N + N*N) )
# Type of neuron: Leaky-integrate-and-fire (LIF)
#
########################################################################################################################

def build_LIF_network(
                        N=500,
                        dt=0.1,
                        time_start = 0.0,
                        time_total=800.0,
                        synapse_density=0.03,
                        g_syn_max = 1,
                        synapse_delay_delta_t = 7.0,
                        tau_syn = 10, # time constant for synaptic g between neurons
                        E_syn_excitatory = 15,
                        E_syn_inhibitory = 0,
                        ei_threshold = 0.85,
                        use_STDP=False,  # Control whether STDP is used to adapt synaptic weights or not
                        tau_W = 3,  # ms (time constant for weight changes in STDP)
                        STDP_scaling = 10.0,
                        state_random_std_dev_noise = 0.4,
                        R=1,
                        C = 25,  # capacitance; larger C leads to smaller effect of stimulus on a neuron's voltage
                        threshold = 10,
                        V_reset = 0,
                        refractory_time = 4,  # ms
                        # Synaptic weight bounds (dimensionless)
                        l_bound = 0,
                        u_bound = 5,
                        number_of_stored_most_recent_spikes = 15,
                        times_array = None,
                        network_name = None
    ):
    """
    Args:
        # current_object
        # Number of neurons
        # N = 500
        # dt = 0.1 # ms
        # time_total = 800.0 # ms
        # Synapse density (1 = fully connected, 0 = never any connection)
        # synapse_density = 0.03
        # Synaptic conductance scaling factor
        # g_syn_max = 1
        # Delay between presynaptic neuron firing and effect on postsynaptic neuron feeling effect
        # synapse_delay_delta_t = 7.0 #ms
        # Synaptic time constant
        # tau_syn = 10 # ms
        # Synaptic Nernst potentials
        # Each presynaptic neuron in this simulation is either inhibitory or excitatory (also known as Dale's Law)
        # Not totally necessary but I'll implement it here anyway
        # E_syn_excitatory = 15 # arbitrarily decided values
        # E_syn_inhibitory = 0
        # ei_threshold = 0.85 # "excite-inhibit threshold". A number between 0 and 1. Percentage of connections which are inhibitory
        # STDP-related variables
        # use_STDP = False,  # Control whether STDP is used to adapt synaptic weights or not
        # tau_W = 3,  # ms
        # STDP_scaling = 10.0,
        # Noise: Standard deviation of noise in V,w initial conditions
        state_random_std_dev_noise = 0.4
        ########################################################################################################################
        # LIF parameters
        ########################################################################################################################
        R = 1
        C = 25 # capacitance; larger C leads to smaller effect of stimulus on a neuron's voltage
        threshold = 10
        V_reset = 0
        refractory_time = 4 # ms
        ########################################################################################################################
        # Synapses
        ########################################################################################################################
        # Synaptic weight bounds (dimensionless)
        l_bound = 0
        u_bound = 5
    """
    # Synaptic weight bounds (dimensionless)
    stats_scale = u_bound - l_bound # used for "scale" argument in data_rvs argument of scipy sparse random method


    E_syn = np.zeros((N))
    for n in range(N):
        random_num = np.random.uniform(0,1)
        E_syn[n] = (random_num<ei_threshold)*E_syn_inhibitory \
                   + (random_num>=ei_threshold)*E_syn_excitatory

    ########################################################################################################################
    # Initial Conditions and Preparing to Solve
    ########################################################################################################################
    # Initial conditions
    # For number N neurons, make all neurons same initial conditions (noise optional):
    state_initial_V_array = np.zeros((N)).astype(float)
    state_initial_V_array = np.multiply(state_initial_V_array,
                                           np.random.normal(loc=1, scale=state_random_std_dev_noise, size=((N)))
                                           )

    # randomly initialize sparse synaptic weights
    # Synaptic connections have shape (N,N), from interval [-1,1)
    matrix_initial_synapse_states = synaptic_weights.make_internal_weights(N, synapse_density, l_bound, stats_scale)
    # bogus stand-in spike time to be replaced
    bogus_spike_time = -10000
    last_firing_times = bogus_spike_time*np.ones((N)) # large negative number so it isn't considered firing at beginning

    # Combine Vnmh and synaptic initial conditions into a single vector
    flattened_initial_V_states = state_initial_V_array.flatten() # shape(4*N)
    flattened_initial_synapse_states = matrix_initial_synapse_states.flatten() # shape(N*N)

    # Combined shape is (N*(2+N))
    flattened_initial_states = np.concatenate((flattened_initial_V_states, flattened_initial_synapse_states))

    ########################################################################################################################
    # Build system
    ########################################################################################################################
    network_1 = LIF_network(flattened_initial_states, times_array, N, R, C, threshold,
                            last_firing_times, V_reset, refractory_time, g_syn_max,
                            E_syn, tau_syn, use_STDP, STDP_scaling, tau_W,
                            synapse_delay_delta_t, number_of_stored_most_recent_spikes, bogus_spike_time,
                            network_name)

    return network_1