from model_PST_LIF_1 import LIF_network
from neuronal_plotting import plot_many_neurons_simultaneous
from neuronal_plotting import make_raster_plot
from spike_methods import spike_list_to_array
import externally_provided_currents as currents
import numpy as np
from scipy.integrate import odeint
import time as time
import scipy.sparse
from scipy import stats

########################################################################################################################
#
# Runs N neurons and N*N synapses in parallel ( system with dimensions (2*N + N*N) )
# Type of neuron: Leaky-integrate-and-fire (LIF)
#
########################################################################################################################

def run_LIF_network(current_object,
                    N=500,
                    dt=0.1,
                    time_start = 0.0,
                    time_total=800.0,
                    synapse_density=0.03,
                    g_syn_max = 1,
                    synapse_delay_delta_t = 7.0,
                    tau_syn = 10,
                    E_syn_excitatory = 15,
                    E_syn_inhibitory = 0,
                    ei_threshold = 0.85,
                    use_STDP=False,  # Control whether STDP is used to adapt synaptic weights or not
                    tau_W = 3,  # ms
                    STDP_scaling = 10.0,
                    ):
    """
    Args:
        # current_object
        # Number of neurons
        # N = 500
        # dt = 0.1 # ms
        # time_total = 800.0 # ms
        #     # Synapse density (1 = fully connected, 0 = never any connection)
        #     synapse_density = 0.03
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
    """

    # Combinations that max RAM if using W_t = (timesteps,N,N):
    # (N=1100, time_total=50)
    # (N=600,  time_total=200)
    # (N=360,  time_total=500)

    # Combinations that max RAM if using W = (N,N):
    # (N=1100, time_total=50)
    # (N=600,  time_total=200)
    # (N=360,  time_total=500)

    timesteps = int(float(time_total)/dt) # total number of intervals to evaluate solution at
    times_array = np.linspace(time_start, time_start + time_total, timesteps)
    print(np.shape(times_array))
    external_current = current_object.function
    # extra descriptors for file name and sometimes plot titles; often contains current name
    extra_descriptors = current_object.name + ';'+current_object.extra_descriptors

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
    state_initial_synaptic = scipy.sparse.random(N,N, density = synapse_density,
                                                 data_rvs=scipy.stats.uniform(loc=l_bound, scale=stats_scale).rvs)
    # bogus stand-in spike time to be replaced
    bogus_spike_time = -10000
    last_firing_times = bogus_spike_time*np.ones((N)) # large negative number so it isn't considered firing at beginning

    # List for spike times to be recorded to
    spike_list = [[] for i in range(N)] # Need N separate empty lists to record spike times to
    last_firing_array = np.zeros((times_array.shape[0], N))

    # Combine Vnmh and synaptic initial conditions into a single vector

    flattened_initial_V_states = state_initial_V_array.flatten() # shape(4*N)
    # remove diagonal synaptic weights
    matrix_initial_synapse_states = scipy.sparse.csr_matrix(state_initial_synaptic).toarray()
    np.fill_diagonal(matrix_initial_synapse_states, val = 0)
    flattened_initial_synapse_states = matrix_initial_synapse_states.flatten() # shape(N*N)

    # Combined shape is (N*(2+N))
    flattened_initial_states = np.concatenate((flattened_initial_V_states, flattened_initial_synapse_states))
    print(flattened_initial_states)

    ########################################################################################################################
    # Build system
    ########################################################################################################################
    network_1 = LIF_network(flattened_initial_states, times_array, N, external_current, R, C, threshold,
                             last_firing_times, V_reset, refractory_time, g_syn_max,
                             E_syn, tau_syn, spike_list, use_STDP, STDP_scaling, tau_W,
                             synapse_delay_delta_t)

    ########################################################################################################################
    # Solve system
    ########################################################################################################################
    # Sol has dimension (time, {state_vars + synapse vars} = {N + N * N} )
    print("Running "+str(N)+" neurons for "+str(timesteps)+" timesteps of size "+str(dt))
    start_time = time.time()
    # sol will contain solutions to system and Euler integration will fill spike_list
    sol_V_t, W_final, last_firing_array= network_1.integrate_forward_LIF_network(dt)
    print("Program took "+str(time.time()-start_time)+" seconds to run.")

    ########################################################################################################################
    # Saving data
    ########################################################################################################################
    spike_list_array = spike_list_to_array(N, spike_list) # array dimension(N, ???)
    np.savetxt('spike_data/spike_list;'+extra_descriptors+'.txt', spike_list_array, fmt='%.3e')
    # dims(timesteps, N):
    np.savetxt('voltages/V;STDP='+str(use_STDP)+';'+str(extra_descriptors)+'.txt',
               sol_V_t.reshape(timesteps, 1, N)[:, 0, :], fmt='%.3e')
    np.savetxt('modified_weights/W;STDP='+str(use_STDP)+';'+str(extra_descriptors)+'.txt',
               W_final, fmt='%.3e')

    # # Plot spike times
    print("SPIKE LIST: "+str(spike_list))
    for n in range(N):
        sorted_unique_last_firing_of_neuron_n = np.unique(last_firing_array[:, n])
        for a_firing_time in sorted_unique_last_firing_of_neuron_n:
            if a_firing_time not in spike_list[n]:
                if a_firing_time != bogus_spike_time:
                    spike_list[n].append(a_firing_time)

    return [sol_V_t, W_final, times_array, timesteps, spike_list, extra_descriptors]

np.random.seed(2021)

N = 500
use_STDP = False

# Imported current
# current_object = currents.I_flat(magnitude=0.5)
# current_object = currents.I_flat_random_noise(magnitude=10, density=0.3)
# current_object = currents.I_flat_random_targets(N, magnitude=10, density=0.1)
# current_object = currents.I_sine(I_max=30)
steps_height_list = [5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
driving_current = currents.multiply_multi_current_object(
    [currents.I_flat_alternating_steps(magnitude=3, I_dt=50, steps_height_list=steps_height_list),
     currents.I_flat_random_targets(N, magnitude=1.0, density=0.1)])

params = {
        # current_object
        'current_object': driving_current,
        # Number of neurons
        'N' : 500,
        'dt' : 0.1, # ms
        'time_total' : 800.0, # ms
        # Synapse density (1 = fully connected, 0 = never any connection)
        'synapse_density' : 0.03,
        # Synaptic conductance scaling factor
        'g_syn_max' : 1,
        # Delay between presynaptic neuron firing and effect on postsynaptic neuron feeling effect
        'synapse_delay_delta_t' : 7.0, #ms
        # Synaptic time constant
        'tau_syn' : 10, # ms
        # Synaptic Nernst potentials
        # Each presynaptic neuron in this simulation is either inhibitory or excitatory (also known as Dale's Law)
        # Not totally necessary but I'll implement it here anyway
        'E_syn_excitatory' : 15, # arbitrarily decided values
        'E_syn_inhibitory' : 0,
        'ei_threshold' : 0.85, # "excite-inhibit threshold". A number between 0 and 1. Percentage of connections which are inhibitory
        # STDP-related variables
        'use_STDP' : False,  # Control whether STDP is used to adapt synaptic weights or not
        'tau_W' : 3,  # ms
        'STDP_scaling' : 10.0
}

results = run_LIF_network(**params)
sol_V_t, W_final, times_array, timesteps, spike_list, extra_descriptors = results

# Plot the active neurons
make_raster_plot(N, spike_list, use_STDP, extra_descriptors)
# Plot the active neurons
plot_many_neurons_simultaneous(N, times_array, sol_V_t.reshape(timesteps, 1, N), use_STDP,
                               extra_descriptors)
