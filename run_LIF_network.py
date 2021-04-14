from model_LIF_network import LIF_network
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

np.random.seed(2021)

# Number of neurons
N = 1000

# Timekeeping (units in milliseconds)
dt = 0.05
time_start = 0.0
time_total = 50.0
timesteps = int(float(time_total)/dt) # total number of intervals to evaluate solution at
times_array = np.linspace(time_start, time_start + time_total, timesteps)
print(np.shape(times_array))
# Imported current
current_object = currents.I_flat(magnitude=0.5)
external_current = current_object.function
# extra descriptors for file name and sometimes plot titles; often contains current name
extra_descriptors = current_object.name + ';'+current_object.extra_descriptors

# Noise: Standard deviation of noise in V,w initial conditions
state_random_std_dev_noise = 0.4

########################################################################################################################
# LIF parameters
########################################################################################################################

R = 1
C = 10
threshold = 10
V_reset = 0
refractory_time = 4 # ms

########################################################################################################################
# Synapses
########################################################################################################################
# Synaptic weight bounds (dimensionless)
l_bound = 0
u_bound = 10
stats_scale = u_bound - l_bound # used for "scale" argument in data_rvs argument of scipy sparse random method

# Synapse density (1 = fully connected, 0 = never any connection)
synapse_density = 1

# Synaptic conductance scaling factor
g_syn_max = 0.5

# Synaptic time constant
tau_syn = 3 # ms

# Synaptic Nernst potentials
# Each presynaptic neuron in this simulation is either inhibitory or excitatory (also known as Dale's Law)
# Not totally necessary but I'll implement it here anyway
E_syn_excitatory = 120 # arbitrarily decided values
E_syn_inhibitory = -50
ei_threshold = 0.8# "excite-inhibit threshold". A number between 0 and 1. Percentage of connections which are inhibitory
E_syn = np.zeros((N))
for n in range(N):
    random_num = np.random.uniform(0,1)
    E_syn[n] = (random_num<ei_threshold)*E_syn_inhibitory \
               + (random_num>=ei_threshold)*E_syn_excitatory

# STDP-related variables
use_STDP = True # Control whether STDP is used to adapt synaptic weights or not
tau_W = 3 # ms
STDP_scaling = 0.1

########################################################################################################################
# Initial Conditions and Preparing to Solve
########################################################################################################################
# Initial conditions
# For number N neurons, make all neurons same initial conditions (noise optional):
# state_initial_V_single = np.array([0.0])
state_initial_V_array = np.zeros((N)).astype(float)
# for i in range(N):
#     # has shape (N)
#     state_initial_V_array[i]=state_initial_V_single
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
flattened_initial_synapse_states = (scipy.sparse.csr_matrix(state_initial_synaptic).toarray()).flatten() # shape(N*N)

# Combined shape is (N*(2+N))
flattened_initial_states = np.concatenate((flattened_initial_V_states, flattened_initial_synapse_states))
print(flattened_initial_states)
########################################################################################################################
# Solve system
########################################################################################################################
# Sol has dimension (time, {state_vars + synapse vars} = {N + N * N} )
print("Running "+str(N)+" neurons for "+str(timesteps)+" timesteps of size "+str(dt))
start_time = time.time()
# sol will contain solutions to system and odeint will fill spike_list
# sol = LIF_network(flattened_initial_states, times_array,
#                   args = (dt, N, external_current, R, C, threshold, last_firing_times, V_reset, refractory_time))
sol_V_t, sol_W_t, last_firing_array= LIF_network(flattened_initial_states, times_array,
                                           dt, N, external_current, R, C, threshold,
                                           last_firing_times, V_reset, refractory_time, g_syn_max,
                                           tau_syn, spike_list, use_STDP)
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
           sol_W_t[-1, :].reshape((N,N)), fmt='%.3e')

# # Plot spike times
print("SPIKE LIST: "+str(spike_list))
for n in range(N):
    for time_index in range(timesteps):
        if last_firing_array[time_index, n] not in spike_list[n]:
            if last_firing_array[time_index, n] != bogus_spike_time:
                # print(last_firing_array[time_index, n])
                spike_list[n].append(last_firing_array[time_index, n])
make_raster_plot(N, spike_list, use_STDP, extra_descriptors)
# Plot the active neurons
plot_many_neurons_simultaneous(N, times_array, sol_V_t.reshape(timesteps, 1, N), use_STDP,
                               extra_descriptors)