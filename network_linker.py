import full_network
import numpy as np
import externally_provided_currents as currents
from model_PST_LIF_1_testing_mods import LIF_network
from neuronal_plotting import plot_many_neurons_simultaneous
from neuronal_plotting import make_raster_plot
from spike_methods import spike_list_to_array

import time as time



# Pseudocode of how I want this to look
np.random.seed(2021)


N_generic = 500
use_STDP = False

# Import observed "sound" data, 1-D. If not imported, then use artificial data
steps_height_list = [5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
current_object = currents.multiply_multi_current_object(
    [currents.I_flat_alternating_steps(magnitude=3, I_dt=50, steps_height_list=steps_height_list),
     currents.I_flat_random_targets(N_generic, magnitude=1.0, density=0.1)])

external_current_function = current_object.function
# extra descriptors for file name and sometimes plot titles; often contains current name
extra_descriptors = current_object.name + ';' + current_object.extra_descriptors

data_1 = external_current_function
data_2 = external_current_function
data_3 = external_current_function

# Convert sound data to F.T.



# Time stuff
dt = 0.1
time_start = 0.0
time_total = 800.0
timesteps = int(float(time_total) / dt)  # total number of intervals to evaluate solution at
times_array = np.linspace(time_start, time_start + time_total, timesteps)
bogus_spike_time = -10000 # time for last_firing_times; large negative number so it isn't considered firing at beginning

########################################################################################################################
# LIF parameters
########################################################################################################################
args_123 = {
    'N' : N_generic,
    'use_STDP' : False,
    # current_object
    'current_object': current_object,
    # Number of neurons
    'N': 500,
    'dt': 0.1,  # ms
    'time_total': 800.0,  # ms
    # Synapse density (1 = fully connected, 0 = never any connection)
    'synapse_density': 0.03,
    # Synaptic conductance scaling factor
    'g_syn_max': 1,
    # Delay between presynaptic neuron firing and effect on postsynaptic neuron feeling effect
    'synapse_delay_delta_t': 7.0,  # ms
    # Synaptic time constant
    'tau_syn': 10,  # ms
    # Synaptic Nernst potentials
    # Each presynaptic neuron in this simulation is either inhibitory or excitatory (also known as Dale's Law)
    # Not totally necessary but I'll implement it here anyway
    'E_syn_excitatory': 15,  # arbitrarily decided values
    'E_syn_inhibitory': 0,
    'ei_threshold': 0.85,
    'last_firing_times' : bogus_spike_time*np.ones((N_generic)),
    # "excite-inhibit threshold". A number between 0 and 1. Percentage of connections which are inhibitory
    # STDP-related variables
    'use_STDP': use_STDP,  # Control whether STDP is used to adapt synaptic weights or not
    'tau_W': 3,  # ms
    'STDP_scaling': 10.0
}

args_mix = args_123.copy()

# Build network objects
network_1 = LIF_network(**args_123)#, STDP_method = STDP_function_interior_1)
network_2 = LIF_network(**args_123)#, STDP_method = STDP_function_interior_1)
network_3 = LIF_network(**args_123)#, STDP_method = STDP_function_interior_1)
network_mix = LIF_network(**args_mix)#, STDP_method = STDP_function_interior_1)

# Create a "full network" object to hold and connect the subnetworks
total_network = full_network()

total_network.set_subnetworks({'network_1' : network_1,
                               'network_2' : network_2,
                               'network_3' : network_3,
                               'network_mix': network_mix
                               })

########################################################################################################################
# Connect networks to input stimulus
########################################################################################################################
# total_network.connect_to_input_data(data_x, network_y); within the total network, uses data_x as stimulus to network_y
total_network.connect_to_input_data(data_1, network_1)
total_network.connect_to_input_data(data_2, network_2)
total_network.connect_to_input_data(data_3, network_3)

########################################################################################################################
# Connect networks to networks, with STDP where necessary
########################################################################################################################

# connect_two_networks(network_x, network_y, STDP_method = STDP_function_z); within the total network,
# connects network_x (presyn) to network_y (postsyn), with STDP_function_z
total_network.connect_two_networks(network_1, network_mix, STDP_method = STDP_function_1_mix)
total_network.connect_two_networks(network_2, network_mix, STDP_method = STDP_function_2_mix)
total_network.connect_two_networks(network_3, network_mix, STDP_method = STDP_function_3_mix)

########################################################################################################################
# Solve system
########################################################################################################################
# Run all networks simultaneously (with STDP on or off) and output results
print("Running " + str(N_generic) + " neurons for " + str(timesteps) + " timesteps of size " + str(dt))
start_time = time.time()
total_network.run(times_array, dt)
network_1_results_unprocessed, network_2_results_unprocessed, \
network_3_results_unprocessed, network_mix_results_unprocessed = total_network.organize_results()
print("Program took " + str(time.time() - start_time) + " seconds to run.")


def process_spike_results(network_result_last_firing_array, N = None): \
    # List for spike times to be recorded to
    spike_list = [[] for i in range(N)]  # Need N separate empty lists to record spike times to

    last_firing_array = network_result_last_firing_array
    spike_list_array = spike_list_to_array(N, spike_list) # array dimension(N, ???)
    np.savetxt('spike_data/spike_list;'+extra_descriptors+'.txt', spike_list_array, fmt='%.3e')
    # dims(timesteps, N):

    # # Plot spike times
    print("SPIKE LIST: "+str(spike_list))
    for n in range(N):
        sorted_unique_last_firing_of_neuron_n = np.unique(last_firing_array[:, n])
        for a_firing_time in sorted_unique_last_firing_of_neuron_n:
            if a_firing_time not in spike_list[n]:
                if a_firing_time != bogus_spike_time:
                    spike_list[n].append(a_firing_time)
    return spike_list


# Process spike results into a more manageable form:
spike_list_network_mix = process_spike_results(network_mix_results_unprocessed[2], N = N_generic)
sol_V_t = network_mix_results_unprocessed[0]
W_final = network_mix_results_unprocessed[1]

# Save voltage and weights
np.savetxt('voltages/V;STDP=' + str(use_STDP) + ';' + str(extra_descriptors) + '.txt',
           sol_V_t.reshape(timesteps, 1, N_generic)[:, 0, :], fmt='%.3e')
np.savetxt('modified_weights/W;STDP=' + str(use_STDP) + ';' + str(extra_descriptors) + '.txt',
           W_final, fmt='%.3e')

# Plot results from network_mix:
# Plot the active neurons
make_raster_plot(N_generic, spike_list_network_mix, use_STDP, extra_descriptors)
# Plot the active neurons
plot_many_neurons_simultaneous(N_generic, times_array, sol_V_t.reshape(timesteps, 1, N_generic), use_STDP,
                               extra_descriptors)
