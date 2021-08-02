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
import numpy.linalg as LA


# random.seed(2021)
class memory_gradient():
    def __init__(self, barrier_height_scaling = 1.0, sigma_outer = 1.0):
        self.solutions_list = []
        self.solutions_array = None
        self.barrier_height_scaling = barrier_height_scaling # overall scale
        self.sigma_outer = sigma_outer # outer wall dropoff rate

    def store_trajectory(self, one_trajectory, dt):
        print("TRAJECTORY HAS SHAPE "+str(np.shape(one_trajectory)))
        # store previous trajectories
        self.solutions_list.append(one_trajectory)
        # turn all x(t) from all trajectories into one array
        # find necessary array size
        needed_length = 0
        for an_x_array in self.solutions_list:
            elements_in_this_array = np.shape(an_x_array)[0]*np.shape(an_x_array)[1]
            needed_length += elements_in_this_array
        self.solutions_array = np.zeros((needed_length))
        # Now add all points from all trajectories into the solutions list
        solutions_array_temp_list = []
        for a_solution_index in range(len(self.solutions_list)):
            an_x_array = self.solutions_list[a_solution_index]
            for t_index, x_t in enumerate(an_x_array):
                solutions_array_temp_list.append(x_t)
        # make list into array
        self.solutions_array = np.array(solutions_array_temp_list)
        print("SHAPE OF SAVED SOLUTIONS IS (SPACE< TIME)")

        self.xdot_array = np.gradient(self.solutions_array, dt, axis = 1)


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

def find_nearest_neighbor(set_of_points, new_point):
    """
    :param set_of_points: ndarray shape (space, timestep_index)
    :param new_point: ndarray (space)
    :param radius: float
    :return: index of point in stored_trajectory closest to new_point
    """
    # distance from a third of all trajectory points
    # print(set_of_points.shape)
    d = LA.norm(set_of_points.transpose() - new_point, axis=1)
    # https://numpy.org/doc/stable/reference/generated/numpy.argmin.html
    ind = np.unravel_index(np.argmin(d, axis=None), d.shape)
    return ind

def find_nearest_neighbor_single_neuron(set_of_points, new_point, neighbor_list=None):
    """
    :param set_of_points: ndarray shape (space, timestep_index)
    :param new_point: ndarray (space)
    :param radius: float
    :return: index of point in stored_trajectory closest to new_point
    """
    # distance from a third of all trajectory points
    # print(set_of_points.shape)
    # d = LA.norm(set_of_points.transpose() - new_point, axis=1)
    # print("SHAPE IS " + str(set_of_points.shape))
    transposed_set_of_points = set_of_points.transpose()
    if neighbor_list is None:
        d = LA.norm(set_of_points.transpose() - new_point, axis=1)
    else:
        # print("NEIGHBOR LIST: " + str(neighbor_list))
        d = np.zeros(transposed_set_of_points.shape)
        for n in range(len(neighbor_list)):  # for all neurons
        # find the nearest neighbor using only that neuron's own neighbors

            d = LA.norm(transposed_set_of_points[:, neighbor_list] - new_point, axis=1)
    # https://numpy.org/doc/stable/reference/generated/numpy.argmin.html
    ind = np.unravel_index(np.argmin(d, axis=None), d.shape)
    return ind

class continuous_network():

    def __init__(self, N, density, input_density, gamma, beta):
        self.N = N # number of nodes
        self.density = density # density of adjacency matrix
        self.input_density = input_density
        self.gamma_1 = gamma # time constant of decay for each neuron; array
        self.beta = beta # sensitivity to input (both external to network or from other nodes)
        self.A = scipy.sparse.random(N, N, density = self.density)# adjacency matrix
        vals, vecs = linalg.eigs(self.A)
        largest_eigenvalue_magnitude_found = np.absolute(vals).max()
        # print(vals)
        print(largest_eigenvalue_magnitude_found)
        self.A = self.A.multiply(1.0/largest_eigenvalue_magnitude_found)
        self.W_in = scipy.sparse.random(N, 3, density=self.input_density, format='csr')
        self.stored_trajectory = None
        self.network_memory_gradient = memory_gradient() # This can be modified in the main script
        self.presyn_neighbor_list = self.A.tolil().rows
        self.times = [] #temporary
        self.gamma_alpha = 500#.1 #should be slow compared to network
        self.gamma_2 = 0.005
        self.gamma_3 = 0.4
        self.gamma_4 = 10
        self.c = 1
        self.sigma_past = 0.03#0.001
        self.sigma_tunnel = 0.1
        self.sigma_tunnel_removal = 2
        self.recall_speedup = 1

        self.R = 1
        self.nearest_neighbor_graph_list = []
        self.absolute_distances_over_time_list = []
        self.term3times4_list = []


    def f(self, t, x, I_ext_f):
        # x is the state; unpacked (alpha, V) = x
        alpha = x[0]
        r = x[1:]
        # Enforce bounds
        r[r>1]=1
        r[r<-1]=-1
        term_2 = 0
        term_3 = 0
        term_4 = 0
        if t>=30 and self.network_memory_gradient.solutions_array is not None:
            diff_vector_list = [] # list of distance vectors for current r(t)
            norm_vector_array = np.empty((self.N)) # array of magnitudes of distance vectors for current r(t)
            unit_vector_list = [ ] # list of unit vectors for current r(t)
            gradient_vector_array = np.empty((self.N))# (shape n) list of change recorded for neuron n based on subspace location
            nearest_center_list = [] # list of vectors for currently nearest centers to nth component of r(t) in neuron n's input subspace
            for n in range(self.N):
                time_index_nearest = find_nearest_neighbor_single_neuron(self.network_memory_gradient.solutions_array[[self.presyn_neighbor_list[n]]], r[self.presyn_neighbor_list[n]])
                nearest_center_list = self.network_memory_gradient.solutions_array[[self.presyn_neighbor_list[n]],
                                                                                  time_index_nearest]
                diff_vector_list.append((r[[self.presyn_neighbor_list[n]]] -
                                         nearest_center_list)[0])
                norm_vector_array[n] = np.linalg.norm(diff_vector_list[n])
                unit_vector_list.append(np.divide(diff_vector_list[n],norm_vector_array[n]))
                gradient_vector_array[n] = self.network_memory_gradient.xdot_array[n,time_index_nearest]
            # term_2 = self.gamma_2 *  np.multiply(-gradient_vector_array, np.exp(-((norm_vector_array)/self.sigma_past)**2))
            nearest_center = (self.network_memory_gradient.solutions_array[:,find_nearest_neighbor(self.network_memory_gradient.solutions_array[:], r)])[:,0]
            diff_full_network_center = (r-nearest_center)
            dist_to_nearest_center = np.linalg.norm(np.fabs(diff_full_network_center))
            dist_to_nearest_center_per_axis = np.sqrt(diff_full_network_center**2)
            term_2 = self.recall_speedup*np.multiply(gradient_vector_array, dist_to_nearest_center<np.sqrt(self.N*self.sigma_past**2)) #np.sum( np.exp(-((norm_vector_array)/self.sigma_past)**2), axis=1))
            term_3 = self.gamma_3 * (1+np.tanh(-self.R*np.exp(-(dist_to_nearest_center_per_axis/self.sigma_tunnel_removal)**2)))
            term_4 = - self.gamma_4 * (diff_full_network_center/self.sigma_tunnel) *np.exp(-(dist_to_nearest_center_per_axis/self.sigma_tunnel)**2) * (1-0*(dist_to_nearest_center/self.sigma_tunnel)**2)
            # term_4 = 500*5*np.tanh(-dist_to_nearest_center_per_axis/self.sigma_tunnel)#500*5*(diff_full_network_center/self.sigma_tunnel)**3 *np.exp(-(dist_to_nearest_center_per_axis/self.sigma_tunnel)**4) * (1-0*(dist_to_nearest_center/self.sigma_tunnel)**2)
            self.absolute_distances_over_time_list.append(dist_to_nearest_center)
            self.term3times4_list.append(np.linalg.norm(np.multiply(term_3,term_4)))
            print("Terms")
            print(dist_to_nearest_center)
            print(term_3)
            print(term_4[:4])
            # TODO: xdot array above
        print(t)
        dadt = self.gamma_alpha*(- alpha + np.tanh(np.linalg.norm(I_ext_f(3,t))))
        # print(0.5*(1+np.tanh(np.linalg.norm(I_ext_f(3,t)))))
        # print(np.linalg.norm(I_ext_f(3,t)))
        drdt_term_usual = alpha*np.multiply(self.gamma_1,(-r + self.beta*np.tanh(self.A@r + self.W_in@I_ext_f(3,t))))
        drdt_term_sst =  (1-alpha)*self.c*(term_2 + np.multiply(copy.deepcopy(term_3),copy.deepcopy(term_4)))
        # if self.network_memory_gradient.solutions_array is not None:
        #     print("drdt: "+str(drdt_term_sst[:5]))
        return np.concatenate((np.array([dadt]), drdt_term_usual + drdt_term_sst))

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


N = 400 # 1000 preferred
density = 0.03 # network node-to-node
# gamma = 5*np.power(10.0,np.random.uniform(low= -1, high= 2, size=N))
gamma = 1
beta = 0.7
input_density = 0.1 # input signal u(t) to network node
alpha_initial = 1.0

start_time = time.time()
state_initial = np.concatenate((np.array([alpha_initial]), np.random.uniform(low=-1, high=1, size=N)))
t_final = 60
dt = 0.05
times_array = np.arange(0.0, t_final, dt)
times_array_cut = np.arange(25, t_final, dt)

network_PST = continuous_network(N, density, input_density, gamma, beta)
# current_object = epc.I_flat()
# current_object = epc.I_sine()
# current_object = epc.L63_object()
current_object_temp_1 = epc.freeze_time_current_object(epc.L63_object(), (10,20) )
current_object_temp_1.prepare_f(times_array)
current_object_train = epc.multiply_multi_current_object([epc.I_flat_cutoff(cutoff_time=70), current_object_temp_1])
# current_object = epc.freeze_time_current_object(current_object_temp_2, (90,100) )
# current_object.prepare_f(times_array)


# check_gen_synch(N,state_initial,times_array,current_object_train.function,network_PST)
solution = network_PST.run(state_initial, times_array, current_object_train.function) # has a .t and .y member variable
print("Program took " + str(round(time.time() - start_time, 2)) + " seconds to run.")


plt.plot([current_object_train.function(3,t) for t in times_array])
plt.title("Current used")
plt.show()

plt.plot(solution.t,solution.y[0].transpose(), linewidth=3, c='r')
plt.title("Alpha during center test")
plt.ylim((0,1.1))
plt.show()

plt.plot(solution.t,solution.y[1:].transpose())
plt.title("Reservoir Activity (Used for Centers)")
plt.ylim((-1,1))
plt.xlim((25,t_final))
# plt.xlim((times_array_cut[-140],times_array_cut[-1]))
plt.show()





# ============ Testing with memory and noise ==============
print("Running again with memory gradient and no noise. Testing nearest neighbors.")

times_array_cut = np.arange(29, t_final, dt)

# state_initial_noisy = np.concatenate((np.array([alpha_initial]),
#                                       np.tanh(state_initial[1:] + 0*np.random.uniform(low=-1, high=1, size=N))))
print((t_final-times_array_cut[0])/0.05)
state_initial_noisy = solution.y[:,int(-(t_final-times_array_cut[0])/0.05)]
# current_object_noisy = epc.L63_object(noise=0.0) # I usually use noise=1
# current_object_noisy.prepare_f(times_array)
current_object_noisy = epc.multiply_multi_current_object([epc.I_flat_cutoff(cutoff_time=30), current_object_train])

network_PST_noisy = copy.deepcopy(network_PST)
network_PST_noisy.network_memory_gradient.store_trajectory(solution.y[1:], dt)
solution_with_memory = network_PST_noisy.run(state_initial_noisy, times_array_cut, current_object_noisy.function) # has a .t and .y member variable
print("Program took " + str(round(time.time() - start_time, 2)) + " seconds to run.")


print("Finished!")
print("SIZE: "+str(solution_with_memory.y.shape))
plt.plot(solution_with_memory.t,solution_with_memory.y[1:].transpose())
plt.title("Reservoir Activity (Testing Against Centers)")
plt.ylim((-1,1))
plt.xlim((25,t_final))
plt.show()

print("")

print(np.shape(solution_with_memory))
print("Finding nearest neighbors")
start_time = time.time()
if network_PST_noisy.network_memory_gradient.solutions_array is not None:
    for state in solution_with_memory.y.transpose():
        nn = find_nearest_neighbor(network_PST_noisy.network_memory_gradient.solutions_array, state[1:])
        network_PST_noisy.nearest_neighbor_graph_list.append(nn[0])
print("Program took " + str(round(time.time() - start_time, 2)) + " seconds to run.")

plt.plot(solution_with_memory.t,solution_with_memory.y[0].transpose(), linewidth=3, c='r')
plt.title("Alpha during second test")
plt.ylim((0,1.1))
plt.show()

plt.plot(times_array_cut,np.array(network_PST_noisy.nearest_neighbor_graph_list),linewidth = 0.3)
plt.title("Nearest neighbors over time")
plt.ylabel('Stored centers timestep index')
plt.xlabel('Time')
plt.plot()
plt.show()


plt.plot(np.array(network_PST_noisy.absolute_distances_over_time_list),linewidth = 0.5)
plt.title("Absolute Distance from Nearest Center Over Time")
plt.ylabel('Distance')
plt.xlabel('Time')
plt.plot()
plt.show()


plt.plot(np.array(network_PST_noisy.term3times4_list),linewidth = 0.5)
plt.title("Term 3 * 4")
plt.ylabel('Product')
plt.xlabel('Time')
plt.plot()
plt.show()