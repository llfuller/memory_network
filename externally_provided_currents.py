import numpy as np
import scipy
from scipy.integrate import odeint


# audio stuff
from scipy.fftpack import fft
from scipy.fftpack import ifft
import soundfile as sf
from scipy.signal import spectrogram

import matplotlib.pyplot as plt



"""
Each class in this script is a current object.
"""

class multiply_multi_current_object():
    def __init__(self, current_objects_list, set_max = None):
        """
        Args:
            current_objects_list (list): list of current objects
            set_max (float): if not set to None, then any function calculated here will have max absolute value set_max.
        """
        self.current_objects_list = current_objects_list
        self.name = ''
        for a_current_object in current_objects_list:
            self.name += a_current_object.name +','
        self.extra_descriptors = ''
        for a_current_object in current_objects_list:
            self.extra_descriptors += a_current_object.extra_descriptors
        self.set_max = set_max

    def function(self, N, t):
        combined_I_ext = np.ones((N))
        for a_current_object in self.current_objects_list:
            combined_I_ext = np.multiply(combined_I_ext, a_current_object.function(N, t))
            # print(a_current_object.name)
            # if a_current_object.name == 'I_flat_cutoff_reverse':
            #     print(combined_I_ext)
        if self.set_max != None:
            max_abs_value = np.amax(np.fabs(combined_I_ext))
            if max_abs_value!=0:
                combined_I_ext = np.multiply(self.set_max, np.divide(combined_I_ext, max_abs_value))
        return combined_I_ext

class sum_multi_current_object():
    def __init__(self, current_objects_list, set_max = None):
        """
        Args:
            current_objects_list (list): list of current objects
            set_max (float): if not set to None, then any function calculated here will have max absolute value set_max.
        """
        self.current_objects_list = current_objects_list
        self.name = ''
        for a_current_object in current_objects_list:
            self.name += a_current_object.name +','
        self.extra_descriptors = ''
        for a_current_object in current_objects_list:
            self.extra_descriptors += a_current_object.extra_descriptors
        self.set_max = set_max

    def function(self, N, t):
        combined_I_ext = np.zeros((N))
        for a_current_object in self.current_objects_list:
            combined_I_ext += a_current_object.function(N, t)
        if self.set_max != None:
            max_abs_value = np.amax(np.abs(combined_I_ext))
            combined_I_ext = np.multiply(self.set_max, np.divide(combined_I_ext, max_abs_value))
        return combined_I_ext


class freeze_time_current_object():
    def __init__(self, current_object, boundary_pair):
        """
        Args:
            current_objects:
            boundary_pair: expects (time_start_freeze, time_end_freeze)
        """
        self.boundary_pair = boundary_pair
        self.name = ''
        self.name += current_object.name +',freeze('+str(self.boundary_pair)+'),'
        self.current_object = current_object
        self.extra_descriptors = ''
        self.extra_descriptors += current_object.extra_descriptors

    def prepare_f(self, args):
        """
        Used to prepare function in cases like L63 object if the function exists inside that object
        """
        self.current_object.prepare_f(args)

    def function(self, N, t):
        time_initial_of_freeze = self.boundary_pair[0]
        time_final_of_freeze = self.boundary_pair[1]
        if t<time_initial_of_freeze:
            return self.current_object.function(N,t)
        if t>=time_initial_of_freeze and t<time_final_of_freeze:
            return self.current_object.function(N,time_initial_of_freeze)
        if t>=time_final_of_freeze:
            return self.current_object.function(N,t-(time_final_of_freeze-time_initial_of_freeze))

# Currents
class I_flat():
    def __init__(self, magnitude = 30):
        """
        Args:
            magnitude (float): magnitude of current supplied to all neurons at all times
        """
        self.name = "I_flat"
        self.magnitude = magnitude
        self.extra_descriptors = ('magnitude='+str(magnitude)).replace('.','p')

    def function(self,N,t):
        I_ext = self.magnitude*np.ones((N))
        return I_ext

class I_flat_random_targets():
    """
    Stimulates random neurons with density given by argument
    """
    def __init__(self, N, target_array = None, magnitude = 30, density = 0.01):
        self.name = "I_flat_random_targets"
        self.density = density
        # Creating target_array
        self.target_array = target_array
        if target_array == None:
            self.target_array = np.random.rand(N)
            # element is 1 if targeted, 0 if not
            self.target_array[self.target_array > self.density] = 0
            self.target_array[self.target_array > 0] = 1
        self.magnitude = magnitude
        self.extra_descriptors = ('magnitude='+str(magnitude)).replace('.','p')

    def function(self,N,t):
        I_ext = self.magnitude*self.target_array
        # for i, an_el in enumerate(I_ext):
        #         if(an_el!=0):
        #             print((i))
        # print(I_ext[:30])
        # print(np.sum(I_ext))
        return I_ext


class I_flat_random_noise():
    def __init__(self, magnitude = 30, density = 0.01):
        self.name = "I_flat_random_noise"
        self.density = density
        self.magnitude = magnitude
        self.extra_descriptors = ('magnitude='+str(magnitude)).replace('.','p')

    def function(self,N,t):
        I_ext = self.magnitude*np.ones((N))
        random_array = np.random.rand(N)
        random_array -=0.5
        I_ext = np.multiply(I_ext, random_array)
        return I_ext


class I_flat_cutoff():

    def __init__(self, cutoff_time, magnitude = 1):
        self.name = "I_flat_cutoff"
        self.magnitude = magnitude
        self.cutoff_time = cutoff_time
        self.extra_descriptors = ('cutoff='+str(cutoff_time)).replace('.','p')

    def function(self, N,t):
        """
        :param N: Included only because many other functions of current objects need N, and this is standardized
        in code that uses current objects' functions.
        :param t: time (scalar)
        :return: current vector
        """
        # Shortened current meant to drive neurons for a small amount of time
        # and cause them to hopefully complete the signal
        I_ext = self.magnitude*np.ones((N))
        if t>self.cutoff_time:
            I_ext = np.zeros((N))
        return I_ext


class I_flat_cutoff_reverse():

    def __init__(self, cutoff_time, magnitude = 1):
        self.name = "I_flat_cutoff_reverse"
        self.magnitude = magnitude
        self.cutoff_time = cutoff_time
        self.extra_descriptors = ('cutoff='+str(cutoff_time)).replace('.','p')

    def function(self, N,t):
        """
        :param N: Included only because many other functions of current objects need N, and this is standardized
        in code that uses current objects' functions.
        :param t: time (scalar)
        :return: current vector
        """
        # Shortened current meant to play only later parts of signal, not earlier parts
        I_ext = self.magnitude*np.ones((N))
        if t<self.cutoff_time:
            # print("Time is "+str(t))
            # print("Setting all to zero")
            I_ext = np.zeros((N))
        return I_ext

class I_select_spatial_components():

    def __init__(self, num_dims, chosen_dims=[1], I_max=1):
        self.name = "I_flat_3_and_5_only"
        self.num_dims = num_dims
        self.chosen_dims = chosen_dims
        self.I_max = I_max
        self.extra_descriptors = ('spatial_dims'+str(self.chosen_dims))

    def function(self, N, t):
        I_ext = self.I_max * np.zeros((self.num_dims))

        # Make sure the correct neurons (3 and 5) receive stimulus
        for i in self.chosen_dims:
            I_ext[i] = 1
        return I_ext

class I_flat_3_and_5_only():

    def __init__(self):
        self.name = "I_flat_3_and_5_only"

    def function(self, N,t,I_max=30):
        I_ext = I_max * np.ones((N))
        if t<40:
            I_ext = I_max*np.ones((N))
        else:
            I_ext = np.zeros((N))

        # Make sure the correct neurons (3 and 5) receive stimulus
        for i in range(N):
            if i!=3 and i!=5:
                I_ext[i] = -1
        return I_ext

class I_sine():

    def __init__(self, magnitude=30, frequency=0.01, cut_time = None):
        self.name = "I_sine"
        self.magnitude = magnitude
        self.frequency = frequency
        self.omega = frequency*(2*np.pi)
        # extra descriptors for plot titles and file names. Must replace . with p to save files safely:
        self.extra_descriptors = ('I_max='+str(magnitude)+';'+'f='+str(frequency)).replace('.','p')
        self.cut_time = cut_time # time beyond which to cut the signal completely

    def function(self, N, t):
        I_ext = self.magnitude*np.cos(self.omega*t)*np.ones((N))
        if (self.cut_time is not None) and (t >= self.cut_time):
            I_ext = np.zeros((N))
        return I_ext

class I_flat_alternating_steps():
    def __init__(self, magnitude=30, I_dt=100, steps_height_list = [5,5,5,5,0]):
        """
        Args:
            magnitude (float): magnitude of current supplied to all neurons at all times
        """
        self.name = "I_flat_alternating_steps"
        self.magnitude = magnitude
        self.extra_descriptors = ('magnitude='+str(magnitude)).replace('.','p')
        self.steps_height_list = steps_height_list
        self.I_dt = I_dt # time between alternating step heights

    def function(self,N,t):
        steps_height_list = self.steps_height_list
        I_ext = self.magnitude * np.ones((N))
        if t<self.I_dt:
            I_ext *= self.steps_height_list[0]
        for i in range(1,len(steps_height_list)):
            if i*self.I_dt < t and t < (i+1)*self.I_dt:
                I_ext *= self.steps_height_list[i]
        return I_ext



class L63_object():
    def __init__(self, rho=28.0, sigma=10.0, beta=8.0 / 3.0, noise=0):
        self.name = "I_L63"
        self.extra_descriptors = ('rho=' + str(rho) +",sig="+str(sigma)+",beta="+str(beta))
        self.rho = rho
        self.sigma = sigma
        self.beta = beta
        self.state0 = [-3.1, -3.1, 20.7]
        self.noise = noise
        self.stored_state = np.array([])
        self.interp_function = None

    def dfdt(self, state, t):
        """To be used in odeint"""
        # runs forward in time from 0, cannot just compute at arbitrary t
        x, y, z = state  # Unpack the state vector
        added_noise = self.noise*scipy.random.uniform(low=0, high=1, size=3)
        return (1+added_noise)*(self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z)  # Derivatives

    def function(self,N,t):
        return self.interp_function(t)

    def prepare_f(self, times_array):
        """
        This needs to be run before the function 'function' can be used for this object.
        :param times_array: array of times at which to produce solution
        :return: states vector from run at times in times_array
        """
        t = times_array

        states = odeint(self.dfdt, self.state0, t)

        # fig = plt.figure()
        # ax = fig.gca(projection="3d")
        # ax.plot(states[:, 0], states[:, 1], states[:, 2])
        # plt.draw()
        # plt.show()
        self.stored_state = states
        self.interp_function = scipy.interpolate.interp1d(times_array, self.stored_state.transpose(), kind='cubic')

        return states

class wavefile_object():
    def __init__(self, filename_load, filename_save, noise=0, times_array = None,
                 num_timesteps_in_window = 3, magnitude_multiplier = 1, time_scaling = 1,
                 input_dimension = 3):
        self.name = "wave_music"
        self.filename_load = filename_load # string; full path to filename + filename and extension
        self.filename_save = filename_save # string; full path to filename + filename and extension
        self.extra_descriptors = ('')
        self.noise = noise
        self.times_array = times_array
        self.t_final = None
        self.t_initial = None
        self.rate = None
        self.num_frames_for_wanted_seconds = None
        self.framespan = None
        self.num_frames_in_window = input_dimension
        self.data_channel = None
        self.interp_function = None
        self.fft_spectrum_t = None
        self.recovered_data = None
        self.start_frame = None
        self.data = None # Data which is directly imported and which is used as a basis for further reduction
        self.end_frame = None # Warning, this is a larger integer than you'd expect, because it needs to be large enough
        # for interpolation of the function to work over the whole integration interval (t_initial, t_final)

        self.rate_seg_div = 30
        self.magnitude_multiplier = magnitude_multiplier
        self.time_scaling = time_scaling
        self.input_dimension = input_dimension

    def load_wavefile(self):
        self.data, self.rate = sf.read(self.filename_load)
        num_frames = self.data.shape[0]
        # seconds_length_song = num_frames / self.rate
        # rate is frames per second
        # number of seconds to take from song
        self.t_initial = self.times_array[0]
        self.t_final = self.times_array[-1]
        self.num_frames_for_wanted_seconds = (self.t_final - self.t_initial) * self.rate
        self.data_channel = self.data[int(self.t_initial*self.rate):int(self.t_final*self.rate), 0]

    def write_wavefile(self):
        sf.write(file=self.filename_save, data=self.recovered_data, samplerate=1,
                 subtype='PCM_16')

    def forward_FFT(self):
        print("Doing forward FFT")
        self.fft_spectrum_t = np.zeros((int(self.end_frame - self.start_frame), self.num_frames_in_window))
        # print("Shape(fft_spectrum_t):"+str(self.fft_spectrum_t.shape))
        for ind in range(int(self.num_frames_for_wanted_seconds-self.num_frames_in_window)):  # timewindows
            # for each timestep, store amplitudes of each frequency occurring over next num_timesteps_in_window
            # print("Frame index: "+str(ind))
            temp = fft(self.data_channel[ind: ind + self.num_frames_in_window])
            # print("fr_ind is "+str(ind))
            # print("temp shape is "+str(temp.shape))
            # print(temp.shape)
            # print("freqs: "+str(np.fft.rfftfreq(len(temp), d=1./rate)))
            # print(temp.shape)
            self.fft_spectrum_t[ind] = temp

    # invert the FFT
    def inverse_FFT(self, returned_data):
        print("Doing inverse FFT")
        self.recovered_data = np.zeros((self.data_channel.shape))
        for fr_ind in self.framespan[:-(self.num_frames_in_window)]:  # timewindows
            # for each timestep, store amplitudes of each frequency occurring over next num_timesteps_in_window
            temp = np.max(np.real(ifft(returned_data[fr_ind])))
            self.recovered_data[fr_ind] = temp

    def function(self,N,t):
        # print("eval for t="+str(t))
        return self.magnitude_multiplier * (self.interp_function(t/self.time_scaling))

    def prepare_f(self, times_array):
        """
        Prepares frequency space data for interpolation.
        This needs to be run before the function 'function' can be used for this object.
        :param times_array: array of times at which to produce solution
        :return: not applicable
        """
        t = times_array
        self.load_wavefile()
        self.start_frame = int(times_array[0] * self.rate) # frame corresponding to first time in times_array
        self.dt = (self.times_array[1]-self.times_array[0])
        self.end_frame =  int(times_array[-1] * self.rate + 2*self.dt*self.rate) # extra dt bit to make interpolation work well near wanted endpoint
        print("start_frame:" +str(self.start_frame))
        print("end_frame:" +str(self.end_frame))
        self.framespan = np.array(range(self.start_frame, self.end_frame))
        # self.framespan = np.array([fr for fr in range(self.num_frames_for_wanted_seconds)])
        print(self.framespan/self.rate)

        f, self.t, self.Sxx = spectrogram(self.data[:self.end_frame, 0], fs=1, nperseg=int(self.rate / self.rate_seg_div),
                                noverlap=int(self.rate / self.rate_seg_div * 9.5 / 10))

        print(self.Sxx.shape)
        print(self.t.shape)

        plt.pcolormesh(self.t / self.rate, f, np.log10(self.Sxx), shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

        print("See shapes here:")
        print(np.concatenate((np.array([0]),self.t)).shape)
        # print(np.concatenate((np.zeros(self.Sxx.shape[0]),self.Sxx))[self.Sxx.shape[0]//self.input_dimension].transpose().shape)
        print(np.vstack((np.zeros(self.Sxx.shape[0]),self.Sxx.transpose())).shape)

        index_spacing = self.Sxx.shape[0]//self.input_dimension
        list_of_indices = [i*index_spacing for i in range(self.input_dimension)]
        # Ensure spatial length is correct
        assert np.zeros(self.Sxx.shape[0])[list_of_indices].shape[0] == self.input_dimension

        a = np.concatenate((np.array([0]),self.t/self.rate, self.end_frame*np.array([1])))
        b = np.vstack((np.zeros(self.Sxx.shape[0]),
                       self.Sxx.transpose(),
                       np.zeros(self.Sxx.shape[0]) ))[:,list_of_indices].transpose()

        # self.forward_FFT()
        # self.interp_function = scipy.interpolate.interp1d(self.framespan/self.rate, self.fft_spectrum_t.transpose(), kind='cubic')
        self.interp_function = scipy.interpolate.interp1d(a,
                                                          np.log10(np.fabs(b)+1),
                                                          kind='cubic') #TODO: Use tanh(log10())