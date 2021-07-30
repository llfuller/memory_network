import numpy as np
import scipy
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import random as random
import scipy.sparse.linalg as linalg
from scipy import interpolate
import time as time
import copy
import externally_provided_currents as epc

# random.seed(2021)
class memory_gradient():
    def __init__(self, barrier_height_scaling = 1.0, sigma_outer = 1.0):
        self.solutions_list = []
        self.solutions_array = np.array([])
        self.barrier_height_scaling = barrier_height_scaling # overall scale
        self.sigma_outer = sigma_outer # outer wall dropoff rate

    def store_trajectory(self, one_trajectory):
        # store previous trajectories
        self.solutions_list.append(one_trajectory)
        # turn all x(t) from all trajectories into one array
        # find necessary array size
        needed_length = 0
        for an_x_array in self.solutions_list:
            elements_in_this_array = np.shape(an_x_array)[0]*np.shape(an_x_array)[1]
            needed_length += elements_in_this_array
        self.solutions_array = np.zeros((needed_length))
        # Now add all points from all trajectories into the solutions_array
        solutions_array_temp_list = []
        for a_solution_index in range(len(self.solutions_list)):
            an_x_array = self.solutions_list[a_solution_index]
            for t_index, x_t in enumerate(an_x_array):
                solutions_array_temp_list.append(x_t)
        self.solutions_array = np.array(solutions_array_temp_list)


    # def calculate_gradient(self, x, t):
    #     """Calculate gradient at a singular x(t) value"""
    #     diff = self.solutions_array-x
    #     # potential, need to take gradient of this
    #     p =
    #     gradient = np.multiply(np.exp(-(1.0/self.sigma_outer)*(diff)**2),
    #                     self.barrier_height_scaling * diff**2
    #                     )
    #     # has maxima at 1/2*(x(t)+-sqrt(4*sigma_outer^2 + x(t)^2)); two hills with valley in center
    #     return gradient


class continuous_network():

    def __init__(self, N, density, input_density, gamma, beta):
        self.N = N # number of nodes
        self.density = density # density of adjacency matrix
        self.input_density = input_density
        self.gamma = gamma # time constant of decay for each neuron
        self.beta = beta # sensitivity to input (both external to network or from other nodes)
        self.A = scipy.sparse.random(N, N, density = self.density)# adjacency matrix
        vals, vecs = linalg.eigs(self.A)
        largest_eigenvalue_magnitude_found = np.absolute(vals).max()
        # print(vals)
        print(largest_eigenvalue_magnitude_found)
        self.A = self.A.multiply(1.0/largest_eigenvalue_magnitude_found)
        self.W_in = scipy.sparse.random(N, 3, density=self.input_density, format='csr')
        self.network_memory_gradient = memory_gradient() # This can be modified in the main script

    def f(self, t, x, I_ext_f):
        # x is the state
        dxdt = self.gamma*(-x + self.beta*np.tanh(self.A@x + self.W_in@I_ext_f(3,t)))
        return dxdt

    def run(self, state_initial, times_array, I_ext_f):
        print("Running continuous network")
        # states = odeint(self.f, state_initial, times_array)
        start_time = times_array[0]
        final_time = times_array[-1]
        states = scipy.integrate.solve_ivp(fun=self.f, t_span=(start_time, final_time),
                                           y0=state_initial, t_eval=times_array,args=(I_ext_f,))
        return states


def check_gen_synch(N,state_initial,
                    times_array,I_ext_f,network_PST):
    """
    Auxiliary method
    """
    state_initial_noisy = state_initial + np.random.uniform(low=0, high=10, size=N)
    solution_1 = network_PST.run(state_initial, times_array, I_ext_f)  # has a .t and .y member variable
    solution_2 = network_PST.run(state_initial_noisy, times_array, I_ext_f)  # has a .t and .y member variable

    gen_synch_exists = np.allclose(solution_1.y[-100:],solution_2.y[-100:], rtol=0.1, equal_nan=True)
    difference = np.log10(np.fabs(solution_1.y[-100:]-solution_2.y[-100:]).transpose())
    plt.plot(times_array, difference)
    plt.ylabel('log10(difference)')
    plt.title("Gen Synch(?) Difference")
    plt.show()
    if gen_synch_exists:
        print("There is gen synch")
    if not gen_synch_exists:
        print("There is NOT perfect gen synch")


N = 1000
density = 0.03
gamma = 5
beta = 0.7
input_density = 0.1

start_time = time.time()
state_initial = np.random.uniform(low=-1, high=1, size=N)
times_array = np.arange(0.0, 150.0, 0.01)
network_PST = continuous_network(N, density, input_density, gamma, beta)
# current_object = epc.I_flat()
# current_object = epc.I_sine()
# current_object = epc.L63_object()
current_object = epc.freeze_time_current_object(epc.L63_object(), 50,60)
current_object.prepare_f(times_array)


check_gen_synch(N,state_initial,times_array,current_object.function,network_PST)
solution = network_PST.run(state_initial, times_array, current_object.function) # has a .t and .y member variable
print("Finished!")
plt.plot(solution.t,solution.y.transpose())
plt.title("Reservoir Activity")
print("Program took " + str(round(time.time() - start_time, 2)) + " seconds to run.")
plt.show()

plt.plot([current_object.function(N,t) for t in times_array])
plt.title("Current used")
plt.show()



# ============ Testing with memory and noise ==============
print("Running again with memory gradient and noisy input and initial state")
state_initial_noisy = np.tanh(state_initial + np.random.uniform(low=-1, high=1, size=N))
current_object_noisy = epc.L63_object(noise=1)
current_object_noisy.prepare_f(times_array)
network_PST_noisy = copy.deepcopy(network_PST)
network_PST_noisy.network_memory_gradient.store_trajectory(solution.y)
solution_with_memory = network_PST_noisy.run(state_initial_noisy, times_array, current_object_noisy.function) # has a .t and .y member variable


print("Finished!")
plt.plot(solution_with_memory.t,solution_with_memory.y.transpose())
plt.title("Reservoir Activity With Noise")
print("Program took " + str(round(time.time() - start_time, 2)) + " seconds to run.")
plt.show()