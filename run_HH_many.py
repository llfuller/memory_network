from model_HH_many import HH_many
from HH_plotting import plot_many_neurons
import numpy as np
from scipy.integrate import odeint
import time as time

# Number of neurons
N = 100

# Timekeeping (units in milliseconds)
dt = 0.02
time_start = 0.0
time_total = 40.0
timesteps = int(float(time_total)/dt) # total number of intervals to evaluate solution at
times_array = np.linspace(time_start, time_start + time_total, timesteps)

# Currents
def I_flat(N,t):
    return 30*np.ones((N))
# I_flat_array has dims(time, N)
I_flat_array = np.zeros((timesteps, N))
for index, t in enumerate(times_array):
    I_flat_array[index, :] = I_flat(N, t)
# Test. Uncomment to make sure the correct neurons (3 and 5) receive stimulus
# for i in range(N):
#     if i!=3 and i!=5:
#         I_flat_array[:,i] = 0

# Initial conditions
# For number N neurons, make all neurons same initial conditions:
state_initial_single = np.array([11,2,2,2])
state_initial_array = np.zeros((4,N))
print(state_initial_array.shape)
for i in range(N):
    # has shape (4, N)
    state_initial_array[:,i]=state_initial_single

# Solve system
# Sol has dimension (time, state_vars = 4 * N)
print("Running "+str(N)+" neurons for "+str(timesteps)+" timesteps of size "+str(dt))
start_time = time.time()
sol = odeint(HH_many, state_initial_array.flatten(), times_array, args = (I_flat,N,timesteps,times_array))
print("Program took "+str(time.time()-start_time)+" seconds to run.")


plot_many_neurons(N, sol.reshape(timesteps, 4, N))