import numpy as np

def HH_many_coupled(state, t, *args):
    # Hodgkin-Huxley model taken from page 38 & 39 of Izhikevich's book Dynamical Systems in Neuroscience
    # Note Hodgkin and Huxley shifted the membrane potential here by ~65mV to make the resting potential ~ 0mV (0.1mV)
    # alpha_n will evaluate to nan if V = 10 at any point in time (very small chance if not purposefully done).
    # Part of model dealing with synapses drawn from https://neuronaldynamics.epfl.ch/online/Ch3.S1.html
    # This is a (online) book written by Wulfram Gerstner, Werner M. Kistler, Richard Naud and Liam Paninski

    e = 2.718281828459  # Euler's number

    N = args[1]  # number of neurons
    timesteps = args[2]
    times_array = args[3]
    last_firing_times = args[4]
    E_syn = args[5]  # mV; arbitrarily decided
    spike_list = args[6]
    V, n, m, h = state[:N*4].reshape((4,N))  # state variables within a single neuron
    W = state[N*4:].reshape((N,N))  # synapse scaling factors
    # I_instr is Current added into cell by man-made instrument
    # I_instr_t is an array with shape(N)
    I_instr_t = args[0](N, t)  # units microAmps/cm^2
    # for n in range(N):
    for i in range(N):
        if (V[i]>40) and ((t-last_firing_times[i])>12):
            last_firing_times[i] = t
            spike_list[i].append(t)
    # last_firing_times[V>80 and (t-last_firing_times)>12] = t
    # print(last_firing_times)
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
    # print(np.multiply(np.matmul(W,g_syn),(V-E_syn))[:4])
    dVdt = - 1.0/C * (g_k*(n**4)*(V-E_k) + g_Na*(m**3)*h*(V-E_Na) + g_L*(V-E_L) +
                      np.multiply(np.matmul(W,g_syn),(V-E_syn)) - I_instr_t)
    dndt = (n_inf - n)/tau_n
    dmdt = (m_inf - m)/tau_m
    dhdt = (h_inf - h)/tau_h

    # Has shape (4*N)
    dVnmhdt = np.array([dVdt, dndt, dmdt, dhdt]).flatten()
    dSynapsesdt = np.zeros(W.shape).flatten() # no synapse adaptation
    dStatedt = np.concatenate((dVnmhdt, dSynapsesdt))
    return dStatedt