from full_network import full_network
import numpy as np
import externally_provided_currents as currents
from model_PST_LIF_1_testing_mods import LIF_network
from neuronal_plotting import plot_many_neurons_simultaneous
from neuronal_plotting import make_raster_plot
from spike_methods import process_spike_results
from run_PST_LIF_1_testing_mods import build_LIF_network

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

########################################################################################################################
# LIF parameters
########################################################################################################################
args_123 = {
    # current_object
    'current_object': current_object,
    # Number of neurons
    'N': N_generic,
    'dt': 0.1,  # ms
    'time_start' : 0.0, # ms
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
    # "excite-inhibit threshold". A number between 0 and 1. Percentage of connections which are inhibitory
    # STDP-related variables
    'use_STDP': use_STDP,  # Control whether STDP is used to adapt synaptic weights or not
    'tau_W': 3,  # ms
    'STDP_scaling': 10.0,
    'R' : 1,
    'C' : 25,  # capacitance; larger C leads to smaller effect of stimulus on a neuron's voltage
    'threshold' : 10,
    'V_reset' : 0,
    'refractory_time' : 4,  # ms
    # Synaptic weight bounds (dimensionless)
    'l_bound' : 0,
    'u_bound' : 5,
    'times_array' : times_array
}

args_mix = args_123.copy()
# Build network objects
network_1 = build_LIF_network(**args_123)#, STDP_method = STDP_function_interior_1)
# network_2 = build_LIF_network(**args_123)#, STDP_method = STDP_function_interior_1)
# network_3 = build_LIF_network(**args_123)#, STDP_method = STDP_function_interior_1)
# network_mix = build_LIF_network(**args_mix)#, STDP_method = STDP_function_interior_1)

# Create a "full network" object to hold and connect the subnetworks
total_network = full_network()
total_network.set_subnetworks({'network_1' : network_1})

# total_network.set_subnetworks({'network_1' : network_1,
#                                'network_2' : network_2,
#                                'network_3' : network_3,
#                                'network_mix': network_mix
#                                })

########################################################################################################################
# Connect networks to input stimulus
########################################################################################################################
# total_network.connect_to_input_data(data_x, network_y); within the total network, uses data_x as stimulus to network_y
total_network.connect_to_input_data(data_1, network_1)
# total_network.connect_to_input_data(data_2, network_2)
# total_network.connect_to_input_data(data_3, network_3)

########################################################################################################################
# Connect networks to networks, with STDP where necessary
########################################################################################################################

# connect_two_networks(network_x, network_y, STDP_method = STDP_function_z); within the total network,
# connects network_x (presyn) to network_y (postsyn), with STDP_function_z
# total_network.connect_two_networks(network_1, network_mix, STDP_method = STDP_function_1_mix)
# total_network.connect_two_networks(network_2, network_mix, STDP_method = STDP_function_2_mix)
# total_network.connect_two_networks(network_3, network_mix, STDP_method = STDP_function_3_mix)

########################################################################################################################
# Solve system
########################################################################################################################
# Run all networks simultaneously (with STDP on or off) and output results
print("Running " + str(N_generic) + " neurons for " + str(timesteps) + " timesteps of size " + str(dt))
start_time = time.time()
total_network.run(times_array, dt)
print("Program took " + str(time.time() - start_time) + " seconds to run.")

# Process spike results into a more manageable form:
# network_1_results, network_2_results, network_3_results, network_mix_results = total_network.organize_results()
network_1_results = total_network.organize_results()


# V_t_mix, W_mix, times_array_mix, spike_list_mix = network_mix_results
V_t_mix, W_mix, times_array_mix, spike_list_mix = network_1_results[0]

########################################################################################################################
# Saving data
########################################################################################################################
# Save voltage and weights
np.savetxt('voltages/V_mix;STDP=' + str(use_STDP) + ';' + str(extra_descriptors) + '.txt',
           V_t_mix.reshape(timesteps, 1, N_generic)[:, 0, :], fmt='%.3e')
np.savetxt('modified_weights/W_mix;STDP=' + str(use_STDP) + ';' + str(extra_descriptors) + '.txt',
           W_mix, fmt='%.3e')
# np.savetxt('spike_data/spike_list_mix;' + extra_descriptors + '.txt', spike_list_to_array(network_mix.N, spike_list), fmt='%.3e')

# Plot results from network_mix:
# Plot the active neurons
make_raster_plot(N_generic, spike_list_mix, use_STDP, extra_descriptors)
# Plot the active neurons
plot_many_neurons_simultaneous(N_generic, times_array, V_t_mix.reshape(timesteps, 1, N_generic), use_STDP,
                               extra_descriptors)
