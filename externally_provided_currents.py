import numpy as np

# Currents
class I_flat():
    def __init__(self):
        self.name = "I_flat"

    def function(self,N,t,I_max=30):
        I_ext = I_max*np.ones((N))
        return I_ext

class I_flat_incomplete():

    def __init__(self):
        self.name = "I_flat_incomplete"

    def function(self, N,t,I_max=30):
        I_ext = I_max*np.ones((N))
        for i in range(N):
            if i in range(5):
                I_ext[i] = 0
        return I_ext

class I_flat_short():

    def __init__(self):
        self.name = "I_flat_short"

    def function(self, N,t,I_max=30):
        # Shortened current meant to drive neurons for a small amount of time
        # and cause them to hopefully complete the signal
        I_ext = I_max*np.ones((N))
        if t>30:
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

        # Test. Uncomment to make sure the correct neurons (3 and 5) receive stimulus
        for i in range(N):
            if i!=3 and i!=5:
                I_ext[i] = -1
        return I_ext

class I_sine():

    def __init__(self, I_max=30, frequency=0.01, cut_time = None):
        self.name = "I_sine"
        self.I_max = I_max
        self.frequency = frequency
        self.omega = frequency*(2*np.pi)
        # extra descriptors for plot titles and file names. Must replace . with p to save files safely:
        self.extra_descriptors = ('I_max='+str(I_max)+';'+'f='+str(frequency)).replace('.','p')
        self.cut_time = cut_time # time beyond which to cut the signal completely

    def function(self, N, t):
        I_ext = self.I_max*np.cos(self.omega*t)*np.ones((N))
        if (self.cut_time is not None) and (t >= self.cut_time):
            I_ext = np.zeros((N))
        return I_ext

class I_sine_slow():

    def __init__(self):
        self.name = "I_sine_slow"

    def function(self, N,t,I_max=30,omega=0.005*(2*np.pi)):
        I_ext = I_max*np.cos(omega*t)*np.ones((N))
        return I_ext

