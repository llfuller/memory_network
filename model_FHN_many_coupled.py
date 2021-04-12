import numpy as np

def FHN_many_coupled(state, t, *args):
    # FHN model form taken from https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model
    # Also described page 106 & 107 of Izhikevich's book Dynamical Systems in Neuroscience

    e = 2.718281828459  # Euler's number

    N = args[1]  # number of neurons
    timesteps = args[2]
    times_array = args[3]
    last_firing_times = args[4]
    E_syn = args[5]  # mV; arbitrarily decided
    spike_list = args[6]
    use_STDP = args[7] # boolean controlling whether STDP adaptation occurs
    tau_W = args[8] # STDP time scale
    STDP_scaling = args[9]
    V, w = state[:N*2].reshape((2,N))  # state variables within a single neuron
    A = state[N*2:].reshape((N,N))  # synapse scaling factors; should not fall below 0 (always keep >=0)
    # I_instr is Current added into cell by man-made instrument
    # I_instr_t is an array with shape(N)
    I_instr_t = args[0](N, t)  # units microAmps/cm^2
    # for n in range(N):
    for i in range(N):
        if (V[i]>0.5) and ((t-last_firing_times[i])>12):
            last_firing_times[i] = t
            spike_list[i].append(t)
    # Cell parameters
    a = 0.8
    b = 0.7
    R = 0.016666 # this value is set so that the scale of I_instr_t affecting V is the same as in model_HH_many_coupled.py
    tau_w = 1.0/0.08

    # Corresponding to membrane resting potential at V~0.

    # Special presynaptic conductance (belongs to presynaptic cell) g_syn
    g_syn_max = 0.5  # mS/cm^2; arbitrarily decided
    tau_syn = 3 # ms
    g_syn = g_syn_max*np.exp(-(t-last_firing_times)/tau_syn)

    # Equations of motion for state variables (should be shape N)
    dVdt = V - (np.power(V,3))/3 - w + np.multiply(np.matmul(A,g_syn),(V-E_syn)) + R*I_instr_t
    dwdt = 1.0/tau_w*(V + a - b*w)
    print(t)
    dVnmhdt = np.array([dVdt, dwdt]).flatten() # Has shape (2*N) after flattening
    dSynapsesdt_temp = np.zeros(A.shape) # initialize dSynapsesdt. Will not change if STDP not used next
    if use_STDP:
        # Implement adaptation by STDP
        # STDP Method 1:
        timing_difference = np.subtract.outer(last_firing_times, last_firing_times) # positive if pre spiked first
        dSynapsesdt_temp = STDP_scaling \
                           * np.multiply(np.sign(timing_difference), np.exp(-np.abs(timing_difference)/tau_W))
        dSynapsesdt_temp[A==0] = 0

    dSynapsesdt = dSynapsesdt_temp.flatten()
    dStatedt = np.concatenate((dVnmhdt, dSynapsesdt))
    return dStatedt