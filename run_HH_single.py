from model_HH_single import HH_single
from HH_plotting import plot_single_neuron
import numpy as np
from scipy.integrate import odeint

# Timekeeping
dt = 0.02
time_start = 0.0
time_total = 100.0
timesteps = int(float(time_total)/dt)
times_array = np.linspace(time_start, time_start + time_total, timesteps)

# Currents
def I_flat(t):
    return 30

# Initial condition
# Single neuron:
state_initial = np.array([11,2,2,2])

# Solve system
# Sol has dimension (time, state_vars)
sol = odeint(HH_single, state_initial, times_array, args = (I_flat,))

plot_single_neuron(sol)