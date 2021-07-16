import numpy as np

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
        # Shortened current meant to drive neurons for a small amount of time
        # and cause them to hopefully complete the signal
        I_ext = self.magnitude*np.ones((N))
        if t>self.cutoff_time:
            I_ext = np.zeros((N))
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