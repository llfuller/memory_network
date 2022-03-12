from model_HH_many_coupled_optimize_W import HH_many_coupled_optimize_W
from neuronal_plotting import plot_many_neurons_simultaneous
from neuronal_plotting import make_raster_plot
import numpy as np
from scipy.integrate import odeint
import time as time
import scipy.sparse
from scipy import stats

# Deprecated

########################################################################################################################
#
# Runs N neurons and N*N synapses in parallel ( system with dimensions (4*N + N*N) )
# Type of neuron: Hodgkin-Huxley
# WARNING: I do not recommend using this specific code.
########################################################################################################################

np.random.seed(2021)

# Number of neurons
N = 30

# STDP-related variables
# See web page http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity
use_STDP = True # Control whether STDP is used to adapt synaptic weights or not
tau_W = 3 # ms
STDP_scaling = 0.0000000005

# Timekeeping (units in milliseconds)
dt = 0.01
time_start = 0.0
time_total = 20.0
timesteps = int(float(time_total)/dt) # total number of intervals to evaluate solution at
times_array = np.linspace(time_start, time_start + time_total, timesteps)

# Synaptic weight bounds (dimensionless)
l_bound = 0
u_bound = 10
stats_scale = u_bound - l_bound # used for "scale" argument in data_rvs argument of scipy sparse random method

# Synapse density (1 = fully connected, 0 = never any connection)
synapse_density = 0.5

# Synaptic Nernst potentials
# Each presynaptic neuron in this simulation is either inhibitory or excitatory (also known as Dale's Law)
# Not totally necessary but I'll implement it here anyway
E_syn_excitatory = 120 # arbitrarily decided values
E_syn_inhibitory = -10
ei_threshold = 0.8# "excite-inhibit threshold". A number between 0 and 1. Percentage of connections which are inhibitory
E_syn = np.zeros((N))
for n in range(N):
    random_num = np.random.uniform(0,1)
    E_syn[n] = (random_num<ei_threshold)*E_syn_inhibitory \
               + (random_num>=ei_threshold)*E_syn_excitatory

# Noise: Standard deviation of noise in V,n,m,h initial conditions (keep below 0.4 to avoid m,n,h below 0 or above 1)
state_random_std_dev_noise = 0.4

# Currents
def I_flat(N,t):
    I_ext = 30*np.ones((N))
    # if t<40:
    #     I_ext = 30*np.ones((N))
    # else:
    #     I_ext = np.zeros((N))

    # # Test. Uncomment to make sure the correct neurons (3 and 5) receive stimulus
    # for i in range(N):
    #     if i!=3 and i!=5:
    #         I_ext[i] = -1

    return I_ext

# Initial conditions
# For number N neurons, make all neurons same initial conditions (noise optional):
state_initial_Vnmh_single = np.array([6,0.5,0.5,0.5])
state_initial_Vnmh_array = np.zeros((4,N)).astype(float)
for i in range(N):
    # has shape (4, N)
    state_initial_Vnmh_array[:,i]=state_initial_Vnmh_single
state_initial_Vnmh_array = np.multiply(state_initial_Vnmh_array,
                                       np.random.normal(loc=1, scale=state_random_std_dev_noise, size=((4,N)))
                                       )
# Ensure gating variables n,m,h are within bounds [0,1]:
for n in range(N):
    for i in range(4):
        if i > 0:
            if state_initial_Vnmh_array[i,n] < 0:
                state_initial_Vnmh_array[i, n] = 0
            elif state_initial_Vnmh_array[i,n] > 1:
                state_initial_Vnmh_array[i, n] = 1

# randomly initialize sparse synaptic weights
# Synaptic connections have shape (N,N), from interval [-1,1)
state_initial_synaptic = scipy.sparse.random(N,N, density = synapse_density,
                                             data_rvs=scipy.stats.uniform(loc=l_bound, scale=stats_scale).rvs)
W_initial_ndarray = scipy.sparse.csr_matrix(state_initial_synaptic).toarray() # shape(N*N)

last_firing_times = -1000*np.ones((N)) # large negative number so it isn't considered firing at beginning

# List for spike times to be recorded to
spike_list = [[] for i in range(N)] # Need N separate empty lists to record spike times to

# Shape is (N*4)
flattened_initial_Vnmh_states = state_initial_Vnmh_array.flatten() # shape(4*N)
flattened_initial_states = flattened_initial_Vnmh_states

# Solve system
# Sol has dimension (time, {state_vars + synapse vars} = {4 * N} )
print("Running "+str(N)+" neurons for "+str(timesteps)+" timesteps of size "+str(dt))
start_time = time.time()
# sol will contain solutions to system and odeint will fill spike_list
sol = odeint(HH_many_coupled_optimize_W, flattened_initial_states, times_array,
             args = (I_flat, N, timesteps, times_array, last_firing_times, E_syn, spike_list,
                     use_STDP, tau_W, STDP_scaling, W_initial_ndarray))
print("Program took "+str(time.time()-start_time)+" seconds to run.")
sol_matrix_Vnmh_and_synapses = sol.reshape(timesteps, 4*N)
sol_matrix_Vnmh_only = sol_matrix_Vnmh_and_synapses[:,:4*N]

# Turn spike list into a saveable array
most_spikes = 0
for n in range(N):
    for time_index in range(len(spike_list[n])):
        if time_index>most_spikes:
            most_spikes = time_index
spike_list_array = np.zeros((N, most_spikes+1))
for n in range(N):
    for time_index in range(len(spike_list[n])):
        spike_list_array[n,time_index] = spike_list[n][time_index]
np.savetxt('spike_list.txt',spike_list_array, fmt='%.3e')

make_raster_plot(N, spike_list, use_STDP)

# Plot the active neurons
plot_many_neurons_simultaneous(N, times_array, sol_matrix_Vnmh_only.reshape(timesteps, 4, N), use_STDP)