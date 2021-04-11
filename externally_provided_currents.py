import numpy as np

# Currents
def I_flat(N,t,I_max=30):
    I_ext = I_max*np.ones((N))
    return I_ext

def I_flat_3_and_5_only(N,t,I_max=30):
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

def I_sine(N,t,I_max=30,omega=0.01*(2*np.pi)):
    I_ext = I_max*np.cos(omega*t)*np.ones((N))
    return I_ext

