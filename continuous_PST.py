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

# import mp3_functions
from pygame import mixer
def play_wavefile(wavefile_name):
    """
    :param wavefile_name: (str) directory+filename+extension
    :return: N/A
    """
    mixer.music.load(wavefile_name)
    mixer.music.play()


random.seed(2022)
np.random.seed(2022)
test_track_directory = 'C:/Users/Lawson/Google Drive/Research/memory_network/testmusic/'
test_track_name = 'testmusic_1'
# test_track_name = 'testmusic_2'


class memory_gradient():
    def __init__(self, barrier_height_scaling = 1.0, sigma_outer = 1.0):
        self.solutions_list = []
        self.solutions_array = None
        self.barrier_height_scaling = barrier_height_scaling # overall scale
        self.sigma_outer = sigma_outer # outer wall dropoff rate

    def store_trajectory(self, one_trajectory, dt):
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
        # SHAPE OF SAVED SOLUTIONS IS (SPACE, TIME)
        self.xdot_array = np.gradient(self.solutions_array, dt, axis = 1)

def find_nearest_neighbor(set_of_points, new_point):
    """
    :param set_of_points: ndarray shape (space, timestep_index)
    :param new_point: ndarray (space)
    :param radius: float
    :return: index of point in stored_trajectory closest to new_point
    """
    # distance from a third of all trajectory points
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
    # d = LA.norm(set_of_points.transpose() - new_point, axis=1)
    transposed_set_of_points = set_of_points.transpose()
    if neighbor_list is None:
        d = LA.norm(set_of_points.transpose() - new_point, axis=1)
    else:
        d = np.zeros(transposed_set_of_points.shape)
        for n in range(len(neighbor_list)):  # for all neurons
        # find the nearest neighbor using only that neuron's own neighbors
            d = LA.norm(transposed_set_of_points[:, neighbor_list] - new_point, axis=1)
    # https://numpy.org/doc/stable/reference/generated/numpy.argmin.html
    ind = np.unravel_index(np.argmin(d, axis=None), d.shape)
    return ind

class continuous_network():

    def __init__(self, N, density, input_density, gamma, beta, input_dimension):
        self.N = N # number of nodes
        self.density = density # density of adjacency matrix
        self.input_density = input_density
        self.gamma_1 = gamma # time constant of decay for each neuron; array
        self.beta = beta # sensitivity to input (both external to network or from other nodes)
        self.A = scipy.sparse.random(N, N, density = self.density)# adjacency matrix
        vals, vecs = linalg.eigs(self.A)
        largest_eigenvalue_magnitude_found = np.absolute(vals).max()
        print(largest_eigenvalue_magnitude_found)
        self.A = self.A.multiply(1.0/largest_eigenvalue_magnitude_found)
        self.W_in = scipy.sparse.random(N, input_dimension, density=self.input_density, format='csr')
        self.stored_trajectory = None
        self.network_memory_gradient = memory_gradient() # This can be modified in the main script
        self.presyn_neighbor_list = self.A.tolil().rows
        self.times = [] #temporary
        self.gamma_alpha = 500#.1 #should be slow compared to network
        self.gamma_2 = 0.005
        self.gamma_3 = 0.4
        self.gamma_4 = 20
        self.c = 1
        self.sigma_past = 0.01#0.001
        self.sigma_tunnel = 0.2
        self.sigma_tunnel_removal = 2
        self.recall_speedup = 0.2
        self.use_local = True

        self.R = 1
        self.nearest_neighbor_graph_list = []
        self.absolute_distances_over_time_list = []
        self.term3times4_list = []

        self.input_dimension = input_dimension

        self.num_t_ind_consider = 5 # must be odd # number of time indices to consider in calculation of potential
        self.mid_local_t_ind = int(self.num_t_ind_consider//2 + self.num_t_ind_consider%2) - 1


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
        alpha = 1
        if t>=t_start_recall and self.network_memory_gradient.solutions_array is not None:
            alpha = 0
            self.diff_vector_list = [] # list of distance vectors for current r(t)
            self.norm_vector_array = np.empty((self.N, self.num_t_ind_consider)) # array of magnitudes of distance vectors for current r(t)
            norm_vector_array_near_five_later_multiply = 1234256789.0 * np.ones((self.N, self.num_t_ind_consider)) # norms of nearest center and two before and two after
            unit_vector_list = [ ] # list of unit vectors for current r(t)
            gradient_vector_array_local = np.empty((self.N))# (shape n) list of change recorded for neuron n based on subspace location TODO: Maybe make this flexible size
            gradient_vector_array_nonlocal = np.empty((self.N))# (shape n) list of change recorded for neuron n based on full state space location
            nearest_center = (self.network_memory_gradient.solutions_array[:,
                              find_nearest_neighbor(self.network_memory_gradient.solutions_array[:], r)])[:, 0]
            diff_full_network_center = (r - nearest_center)
            dist_to_nearest_center = np.linalg.norm(np.fabs(diff_full_network_center))
            dist_to_nearest_center_per_axis = np.fabs(diff_full_network_center)

            if self.use_local:
                nearest_center_vector_list = []
                self.gamma_3 = 1.6/3.5
                self.sigma_tunnel = 0.35
                self.sigma_past = 0.04
                self.sigma_tunnel_removal = 0

                element_to_set_as_zero = self.mid_local_t_ind*np.ones((self.N)) # for tunnel_removal term (which center to ignore proximity for), by default = 2 for 5
                self.local_displacements = np.empty((self.N, self.num_t_ind_consider))
                for n in range(self.N): # for each node
                    time_index_nearest_local = np.array(find_nearest_neighbor_single_neuron(
                        self.network_memory_gradient.solutions_array[[self.presyn_neighbor_list[n]]],
                        r[self.presyn_neighbor_list[n]]))
                    # # list of vectors for currently nearest centers to nth component of r(t) in neuron n's input subspace
                    # print(" time_index_nearest_local: "+str(time_index_nearest_local))

                    # print("SHape of time_index_nearest_local: "+str(np.array(time_index_nearest_local).shape))
                    time_index_1 = time_index_nearest_local - 2
                    time_index_2 = time_index_nearest_local - 1
                    time_index_3 = time_index_nearest_local
                    time_index_4 = time_index_nearest_local + 1
                    time_index_5 = time_index_nearest_local + 2
                    final_time_index = self.network_memory_gradient.solutions_array.shape[1] - 1 # final valid index for any array with dimension (time)
                    # Account for edge cases
                    if time_index_nearest_local <= self.mid_local_t_ind:
                        time_index_1, time_index_2, time_index_3, time_index_4, time_index_5 = [0,1,2,3,4]
                    if time_index_nearest_local >= final_time_index-self.mid_local_t_ind:
                        time_index_1, time_index_2, time_index_3, time_index_4, time_index_5 = [final_time_index-4, final_time_index-3, final_time_index-2, final_time_index-1, final_time_index]

                    # closest x(center) vector in subspace
                    nearest_center_vector_list.append(self.network_memory_gradient.solutions_array[[self.presyn_neighbor_list[n]],
                                                                                       time_index_nearest_local])

                    # make special cases for element_to_set_as_zero if node is closest to boundary centers
                    if time_index_nearest_local <= self.mid_local_t_ind: # if time index is less than 2 for that
                        element_to_set_as_zero[n] = int(time_index_nearest_local)
                    if time_index_nearest_local >= final_time_index - self.mid_local_t_ind:
                        element_to_set_as_zero[n] = int(4-(final_time_index - time_index_nearest_local))
                    element_to_set_as_zero = element_to_set_as_zero.astype('int')

                    # r(t)-r(local_center)
                    # print(self.network_memory_gradient.solutions_array[n, int(time_index_nearest_local-1):int(time_index_nearest_local+2)])
                    # print("Time indices "+str((int(time_index_nearest_local_m1),int(time_index_nearest_local_p1+1))))
                    self.local_displacements[n,:] = r[n] -self.network_memory_gradient.solutions_array[n,int(time_index_1):int(time_index_5+1)]
                    # print(local_displacement)
                    # local subspace vector to center in subspace
                    # (x(t) - x(local_center))_n
                    self.diff_vector_list.append(np.array([
                        r[[self.presyn_neighbor_list[n]]] -
                        (self.network_memory_gradient.solutions_array[[self.presyn_neighbor_list[n]],
                                                                      time_index_1])[0],
                        r[[self.presyn_neighbor_list[n]]] -
                        (self.network_memory_gradient.solutions_array[[self.presyn_neighbor_list[n]],
                                                                      time_index_2])[0],
                        r[[self.presyn_neighbor_list[n]]] -
                        (self.network_memory_gradient.solutions_array[[self.presyn_neighbor_list[n]],
                                                                      time_index_3])[0],
                        r[[self.presyn_neighbor_list[n]]] -
                        (self.network_memory_gradient.solutions_array[[self.presyn_neighbor_list[n]],
                                                                      time_index_4])[0],
                        r[[self.presyn_neighbor_list[n]]] -
                        (self.network_memory_gradient.solutions_array[[self.presyn_neighbor_list[n]],
                                                                      time_index_5])[0]
                    ]
                    ))
                    # diff_vector_arr = np.array(diff_vector_list) # array of the above list
                    # local subspace distance (cannot take negative values) to center in subspace
                    self.norm_vector_array[n] = np.linalg.norm(self.diff_vector_list[n],axis=1)
                    # print(self.norm_vector_array)
                    # # local subspace unit vector (can take negative values) to center in subspace
                    # unit_vector_list.append(np.divide(diff_vector_list[n], norm_vector_array[n]))

                    # gradient experienced by neuron n in its local subspace due to subspace center
                    gradient_vector_array_local[n] = self.network_memory_gradient.xdot_array[
                        n, time_index_nearest_local]
                    # Note norm_vector_array[n] is the norm to nearest center in subspace n
                    # diff_vector_list contains n arrays (1 per neuron) of shape (5, #local presyn neighbors)
                    # norm_vector_array_near_five_later_multiply[n] = np.linalg.norm(self.diff_vector_list[n], axis=1)
                # norm_vector_array_near_five_later_multiply[:,element_to_set_as_zero] = 1234256789.0 # negate closest center influence by treating it as if it is very far away (dist 1234256789.0)

                # Code for local distance measurement to neuron subspace centers
                # term_2 = self.recall_speedup*np.multiply(gradient_vector_array_local, np.fabs(norm_vector_array)<self.sigma_past) #np.sum( np.exp(-((norm_vector_array)/self.sigma_past)**2), axis=1))
                term_2 =  self.recall_speedup*np.multiply(gradient_vector_array_local[..., None], np.exp(-(self.norm_vector_array/self.sigma_past)**2 - (self.local_displacements/self.sigma_past)**2))
                term_3 = self.gamma_3 #* (1+np.tanh(self.R* np.exp(-(self.norm_vector_array/self.sigma_tunnel_removal)**2)))#[..., None] #* (1+np.tanh(-self.R*np.exp(-(norm_vector_array/self.sigma_tunnel_removal)**2)))
                term_4 = - self.gamma_4 * (self.local_displacements/self.sigma_tunnel) *np.exp(-(self.norm_vector_array/self.sigma_tunnel)**2) #* (1-0*(dist_to_nearest_center/self.sigma_tunnel)**2)
            else:
                pass # not implemented anymore
                # for n in range(self.N):
                #     time_index_nearest_nonlocal = find_nearest_neighbor(self.network_memory_gradient.solutions_array, r)
                #     # gradient experienced by neuron n in total state space due to center
                #     gradient_vector_array_nonlocal[n] = self.network_memory_gradient.xdot_array[n,time_index_nearest_nonlocal]
                # # term_2 = self.gamma_2 *  np.multiply(-gradient_vector_array, np.exp(-((norm_vector_array)/self.sigma_past)**2))
                #
                # # Code for nonlocal distance measurement to network centers
                # term_2 = self.recall_speedup*np.multiply(gradient_vector_array_nonlocal, dist_to_nearest_center<np.sqrt(self.N*self.sigma_past**2)) #np.sum( np.exp(-((norm_vector_array)/self.sigma_past)**2), axis=1))
                # term_3 = self.gamma_3 * (1+np.tanh(-self.R*np.exp(-(dist_to_nearest_center_per_axis/self.sigma_tunnel_removal)**2)))
                # term_4 = - self.gamma_4 * (diff_full_network_center/self.sigma_tunnel) *np.exp(-(dist_to_nearest_center_per_axis/self.sigma_tunnel)**2) * (1-0*(dist_to_nearest_center/self.sigma_tunnel)**2)


            # term_4 = 500*5*np.tanh(-dist_to_nearest_center_per_axis/self.sigma_tunnel)#500*5*(diff_full_network_center/self.sigma_tunnel)**3 *np.exp(-(dist_to_nearest_center_per_axis/self.sigma_tunnel)**4) * (1-0*(dist_to_nearest_center/self.sigma_tunnel)**2)
            self.absolute_distances_over_time_list.append(dist_to_nearest_center)
            self.term3times4_list.append(np.linalg.norm(np.multiply(term_3,term_4)))
            # TODO: xdot array above
        print(t)
        # print(I_ext_f(self.input_dimension,t))
        dadt = 0#self.gamma_alpha*(- alpha + np.tanh(np.linalg.norm(I_ext_f(self.input_dimension,t))))
        # print("new shape: "+str(((1-alpha)*self.c*(term_2 + np.multiply(copy.deepcopy(term_3),copy.deepcopy(term_4)))).shape))
        drdt_term_usual = alpha*np.multiply(self.gamma_1,(-r + self.beta*np.tanh(self.A@r + self.W_in@I_ext_f(self.input_dimension,t))))
        drdt_term_sst = 0
        drdt_term_sst_unsummed = (1-alpha)*self.c*(term_2 + np.multiply(copy.deepcopy(term_3),copy.deepcopy(term_4)))
        if t>=t_start_recall and self.network_memory_gradient.solutions_array is not None:
            drdt_term_sst =  np.sum(drdt_term_sst_unsummed, axis= 1)
        else:
            drdt_term_sst = 0
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

#############################
# Parameters
#############################
N = 1000 # 1000 preferred
density = 0.03 # network node-to-node
# gamma = 5*np.power(10.0,np.random.uniform(low= -1, high= 2, size=N))
gamma = 50#5 #500 works well for songs, 5 works well for L96
beta = 0.7
input_density = 0.01 # input signal u(t) to network node
alpha_initial = 1.0

start_time = time.time()
state_initial = np.concatenate((np.array([alpha_initial]), np.random.uniform(low=-1, high=1, size=N)))
t_initial = 0
t_start_recall = 2
t_final = 3#15

dt = 0.001
times_array = np.arange(t_initial, t_final, dt)

input_dimension = 203#3 # dimension of input current
print("Building network object")
network_PST = continuous_network(N, density, input_density, gamma, beta, input_dimension)
print("Initializing current array")
# current_object_temp_1 = epc.wavefile_object(filename_load = test_track_directory+test_track_name+'.wav',
#                                             filename_save = test_track_directory+test_track_name+'_modified2.wav',
#                                             noise = 0,
#                                             times_array=times_array,
#                                             magnitude_multiplier = 1000,
#                                             time_scaling = 1,
#                                             input_dimension = input_dimension
#                                             )

# current_object_temp_1 = epc.freeze_time_current_object(epc.L63_object(), (10,20) )
# current_object_temp_1 = epc.L63_object()

# =========== 4 Joint System =============
# current_object_temp_1a = epc.L96_object(dims = 4)
# current_object_temp_1b = epc.L63_object()
# current_object_temp_1c = epc.Colpitts_object()
# current_object_temp_1d = epc.NaKL_object()
#
# current_object_temp_1 = epc.append_multi_current_object(current_objects_list=[current_object_temp_1a,
#                                                                               current_object_temp_1b,
#                                                                               current_object_temp_1c,
#                                                                               current_object_temp_1d],
#                                                         current_objs_dims_list=[4,3,3,4])

# joint music with slow Colpitts
current_object_temp_1a = epc.wavefile_object(filename_load = test_track_directory+test_track_name+'.wav',
                                            filename_save = test_track_directory+test_track_name+'_modified3.wav',
                                            noise = 0,
                                            times_array=times_array,
                                            magnitude_multiplier = 1,
                                            time_scaling = 1,
                                            input_dimension = 200
                                            )
current_object_temp_1b = epc.Colpitts_object()
current_object_temp_1 = epc.append_multi_current_object(current_objects_list=[current_object_temp_1a,
                                                                              current_object_temp_1b],
                                                        current_objs_dims_list=[200,3])



current_object_temp_1.prepare_f(times_array)
# plt.plot(current_object_temp_1.framespan/current_object_temp_1.rate, current_object_temp_1.fft_spectrum_t)
# plt.show()

# current_object_train = epc.multiply_multi_current_object([epc.I_flat_cutoff(cutoff_time=70), current_object_temp_1])
current_object_train = current_object_temp_1


# check_gen_synch(N,state_initial,times_array,current_object_train.function,network_PST)
solution = network_PST.run(state_initial, times_array, current_object_train.function) # has a .t and .y member variable

class imported_solution_obj():
    def __init__(self, solution_y, times_array):
        self.y = solution_y
        self.t = times_array
# solution_with_memory = imported_solution_obj(np.load("data_continuous/song+Colpitts_solution_y.npy"),
#                                              np.load("data_continuous/song+Colpitts_times_array.npy")
#                                              )

# print("Program took " + str(round(time.time() - start_time, 2)) + " seconds to run.")
plt.plot(solution.t,solution.y[1:].transpose())
plt.title("Reservoir Activity (Used for Centers)")
plt.ylim((-1,1))
plt.xlim((t_initial,t_final))
# plt.xlim((times_array_cut[-140],times_array_cut[-1]))
plt.show()

def calculate_W_out(r, u, beta):
    """
    :param r: (ndarray) dim(space,time)
    :param u: (ndarray) dim(time, space)
    :param beta: (float) Tikhonov regularization parameter
    :return: W_out matrix that maps from r to u.
    """
    # see Lukosevicius Practical ESN eqtn 11
    # https://en.wikipedia.org/wiki/Linear_regression
    # Using ridge regression
    print(r.shape)
    print(u.shape)
    train_start_timestep = 0
    train_end_timestep = r.shape[1] # length along time axis
    # spatial dimensions
    N_r = scipy.shape(r)[0]
    N_u = scipy.shape(u)[1]

    # R = r[train_start_timestep:train_end_timestep].transpose()
    # U = u[train_start_timestep:train_end_timestep].transpose()
    # Ridge Regression
    print("Shape of scipy.matmul(r, r.transpose()): "+str(scipy.matmul(r, r.transpose()).shape))
    print("Shape of scipy.matmul(r, r.transpose()): "+str(scipy.identity(N_r).shape))

    W_out = scipy.matmul(scipy.array(u[train_start_timestep:train_end_timestep,:].transpose()),
                         scipy.matmul(r.transpose(),
                                      scipy.linalg.inv(
                                          scipy.matmul(r, r.transpose()) + beta * scipy.identity(N_r))))
    return W_out

current_used_to_establish_centers_against_time_array = np.array([current_object_train.function(input_dimension,t) for t in times_array])
W_out = calculate_W_out(r = solution.y[1:],
                        u = current_used_to_establish_centers_against_time_array,
                        beta = 0.5)#0.0001)
# beta = 0.001 for L63 and stuff, and 10 for song

# Current plot broken down
# fig, axis_tuple = plt.subplots(input_dimension, 1)
# fig.suptitle('Current used')
# for ind, an_axis in enumerate(axis_tuple):
#     an_axis.plot(times_array, current_used_to_establish_centers_against_time_array[:,ind])
#     an_axis.set_xlabel('time')
#     an_axis.set_ylabel('dim_'+str(ind))
# plt.show()

# ============ Testing with memory and noise ==============
print("Running again with memory gradient and no noise. Testing nearest neighbors.")

times_array_cut = np.arange(t_initial, t_final, dt)

# state_initial_noisy = np.concatenate((np.array([alpha_initial]),
#                                       np.tanh(state_initial[1:] + 0*np.random.uniform(low=-1, high=1, size=N))))
# state_initial_noisy = solution.y[:,int(t_start_recall/dt)]
# before Sept 13, this state_initial_noisy was same as state_initial
state_initial_noisy =  np.concatenate((np.array([alpha_initial]), np.random.uniform(low=-1, high=1, size=N)))
# # current_object_noisy = epc.L63_object(noise=0.0) # I usually use noise=1
# # current_object_noisy.prepare_f(times_array)
current_object_noisy_temp = epc.multiply_multi_current_object([current_object_train, epc.I_flat_cutoff(cutoff_time=t_start_recall)])
current_object_noisy = epc.multiply_multi_current_object([current_object_noisy_temp, epc.I_select_spatial_components(num_dims=input_dimension, chosen_dims=np.array(range(input_dimension))[:])])
# #TODO: Use time normalized derivative?
network_PST_noisy = copy.deepcopy(network_PST)
network_PST_noisy.network_memory_gradient.store_trajectory(solution.y[1:], dt)
solution_with_memory = network_PST_noisy.run(state_initial_noisy, times_array_cut, current_object_noisy.function) # has a .t and .y member variable
print("Program took " + str(round(time.time() - start_time, 2)) + " seconds to run.")

# ============= Find nearest neighbors at all points in time ======================

print("Finding nearest neighbors and absolute distance to nearest neighbor")
start_time = time.time()
network_PST_noisy.absolute_distances_over_time_list = []
if network_PST_noisy.network_memory_gradient.solutions_array is not None:
    for state in solution_with_memory.y.transpose():
        nn = find_nearest_neighbor(network_PST_noisy.network_memory_gradient.solutions_array, state[1:])
        network_PST_noisy.nearest_neighbor_graph_list.append(nn[0])
        # print(state.shape)
        r = state[1:]
        nearest_center = (network_PST_noisy.network_memory_gradient.solutions_array[:,
                          find_nearest_neighbor(network_PST_noisy.network_memory_gradient.solutions_array[:], r)])[:, 0]
        diff_full_network_center = (r - nearest_center)
        dist_to_nearest_center = np.linalg.norm(np.fabs(diff_full_network_center))
        network_PST_noisy.absolute_distances_over_time_list.append(dist_to_nearest_center)
        # print(nn)
print("NN info gathering took " + str(round(time.time() - start_time, 2)) + " seconds to run.")




# # ###########################
# # # Plotting
# # ###########################
plt.plot(times_array, current_used_to_establish_centers_against_time_array)
plt.title("Current used to establish centers")
plt.ylim((-1,1)) #for music
plt.show()


# plt.plot(solution.t,solution.y[0].transpose(), linewidth=3, c='r')
# plt.title("Alpha during center test")
# plt.ylim((0,1.1))
# plt.show()

plt.plot(times_array,[current_object_noisy.function(input_dimension,t) for t in times_array])
plt.title("Current used to test recall")
# plt.xlim((5,60))
plt.show()

plt.plot(solution_with_memory.t,solution_with_memory.y[1:].transpose())
plt.title("Reservoir Activity (Testing Against Centers)")
plt.ylim((-1,1))
plt.xlim((t_initial,t_final))
plt.show()

# plt.plot(solution_with_memory.t,solution_with_memory.y[0].transpose(), linewidth=3, c='r')
# plt.title("Alpha during second test")
# plt.ylim((0,1.1))
# plt.show()

plt.plot(times_array_cut,np.array(network_PST_noisy.nearest_neighbor_graph_list),linewidth = 0.3)
plt.title("Nearest neighbors over time")
plt.ylabel('Stored centers timestep index')
plt.xlabel('Time')
plt.show()
#
plt.plot(times_array_cut, np.array(network_PST_noisy.absolute_distances_over_time_list),linewidth = 0.5)
plt.title("Absolute Distance from Nearest Center Over Time")
plt.ylabel('Distance')
plt.xlabel('Time')
plt.plot()
plt.show()

plt.plot(times_array, (W_out@solution_with_memory.y[1:]).transpose())
plt.title("Predicted Output")
plt.ylim((-1,1.0)) #for music
# plt.ylim((-25,45)) # for l63
# plt.xlim((5,60))
plt.show()

# #================================

# np.save("data_continuous/song+Colpitts_solution_W_out.npy",W_out)
# np.save("data_continuous/song+Colpitts_solution_y.npy",solution_with_memory.y)
# np.save("data_continuous/song+Colpitts_times_array.npy",times_array)
# np.save("data_continuous/song+Colpitts_nearest_neighbor_graph_list.npy",network_PST_noisy.nearest_neighbor_graph_list)
# np.save("data_continuous/song+Colpitts_absolute_distances_over_time_list.npy",network_PST_noisy.absolute_distances_over_time_list)
# current_object_temp_1a.inverse_FFT(((current_used_to_establish_centers_against_time_array)[:,:-3]).transpose())
# current_object_temp_1a.inverse_FFT((W_out@solution_with_memory.y[1:])[:-3])
# current_object_temp_1a.inverse_FFT((current_used_to_establish_centers_against_time_array.T)[:-3])

# # write music to file and play
# mixer.init()
# current_object_temp_1a.write_wavefile()
# play_wavefile(test_track_directory+test_track_name+'_modified3.wav')
