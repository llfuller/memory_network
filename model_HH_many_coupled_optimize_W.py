import numpy as np

def HH_many_coupled_optimize_W(state, t, *args):
    # Hodgkin-Huxley model taken from page 38 & 39 of Izhikevich's book Dynamical Systems in Neuroscience
    # Note Hodgkin and Huxley shifted the membrane potential here by ~65mV to make the resting potential ~ 0mV (0.1mV)
    # alpha_n will evaluate to nan if V = 10 at any point in time (very small chance if not purposefully done).
    # Part of model dealing with synapses drawn from https://neuronaldynamics.epfl.ch/online/Ch3.S1.html
    # This is a (online) book written by Wulfram Gerstner, Werner M. Kistler, Richard Naud and Liam Paninski
    # WARNING: I do not recommend using this specific code.

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
    W = args[10]  # synapse scaling factors
    V, n, m, h = state[:N*4].reshape((4,N))  # state variables within a single neuron
    # I_instr is Current added into cell by man-made instrument
    # I_instr_t is an array with shape(N)
    I_instr_t = args[0](N, t)  # units microAmps/cm^2
    # for n in range(N):
    for i in range(N):
        if (V[i]>40) and ((t-last_firing_times[i])>12):
            last_firing_times[i] = t
            spike_list[i].append(t)
    # Cell parameters
    C = 1  # Membrane capacitance; microFarads/cm^2
    # Shifted Nernst equilibrium potentials
    E_k = -12  # mV
    E_Na = 120  # mV
    E_L = 10.6  # mV
    # Maximal conductances
    g_k = 36  # mS/cm^2
    g_Na = 120  # mS/cm^2
    g_L = 0.3  # mS/cm^2

    # Corresponding to membrane resting potential at V~0.
    # Rate variables
    alpha_n = 0.01*(10.0-V)/(e**((10.0-V)/10.0)-1.0)
    beta_n = 0.125 * e**(-V/80.0)
    alpha_m = 0.1*(25.0-V)/(e**((25.0-V)/10.0) - 1.0)
    beta_m = 4.0*e**(-V/18.0)
    alpha_h = 0.07*e**(-V/20.0)
    beta_h = 1.0/(e**((30.0-V)/10.0) + 1.0)

    # Gating variables at saturation
    n_inf = alpha_n/(alpha_n + beta_n)
    m_inf = alpha_m/(alpha_m + beta_m)
    h_inf = alpha_h/(alpha_h + beta_h)
    tau_n = 1/(alpha_n + beta_n)
    tau_m = 1/(alpha_m + beta_m)
    tau_h = 1/(alpha_h + beta_h)

    # Special presynaptic conductance (belongs to presynaptic cell) g_syn
    g_syn_max = 0.5  # mS/cm^2; arbitrarily decided
    tau_syn = 3 # ms
    g_syn = g_syn_max*np.exp(-(t-last_firing_times)/tau_syn)


    # Equations of motion for state variables (should be shape N)
    dVdt = - 1.0/C * (g_k*(n**4)*(V-E_k) + g_Na*(m**3)*h*(V-E_Na) + g_L*(V-E_L) +
                      np.multiply(np.matmul(W,g_syn),(V-E_syn)) - I_instr_t)
    dndt = (n_inf - n)/tau_n
    dmdt = (m_inf - m)/tau_m
    dhdt = (h_inf - h)/tau_h

    dVnmhdt = np.array([dVdt, dndt, dmdt, dhdt]).flatten() # Has shape (4*N) after flattening
    dSynapsesdt = np.zeros(W.shape) # initialize dSynapsesdt. Will not change if STDP not used next
    if use_STDP:
        # Implement adaptation by STDP
        # STDP Method 1:
        for i in range(N): # postsynaptic
            if len(spike_list[i]) > 0 and (t-spike_list[i][-1]) < 10:
                for j in range(N): # presynaptic
                    if i!=j and len(spike_list[j]) > 0 and (t-spike_list[j][-1]) < 10: # if both have spiked at all
                        timing_difference = spike_list[i][-1] - spike_list[j][-1] # positive if pre spiked first
                        dSynapsesdt[i,j] = STDP_scaling * np.sign(timing_difference) \
                                                * np.exp(-timing_difference/tau_W) # no synapse adaptation
        # STDP Method 2. Faster alternative, but maybe not a good model of it:
        # timing_difference = np.divide(1.0,(np.abs(np.subtract.outer(dVdt, dVdt))+1.0)) # only as good as it can get if neurons have few connections
        # dSynapsesdt_temp = STDP_scaling*np.sign(timing_difference)*np.exp(-timing_difference/tau_W)
    W += dSynapsesdt
    # for i in range(N):  # postsynaptic
    #     if len(spike_list[i])>0 and (t-spike_list[i,-1]) < 5:
    #             for j in range(N): # presynaptic
    #                 if len(spike_list[j])>0 and (t-spike_list[i,-1]) < 5:
    #                     if
    dStatedt = dVnmhdt
    return dStatedt