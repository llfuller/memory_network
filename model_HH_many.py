import numpy as np

# Deprecated

def HH_many(state, t, *args):
    # Hodgkin-Huxley model taken from page 38 & 39 of Izhikevich's book Dynamical Systems in Neuroscience
    # Note Hodgkin and Huxley shifted the membrane potential here by ~65mV to make the resting potential ~ 0mV (0.1mV)
    # alpha_n will evaluate to nan if V = 10 at any point in time (very small chance if not purposefully done).
    N = args[1]  # number of neurons
    timesteps = args[2]
    times_array = args[3]
    V, n, m, h = state.reshape((4,N))
    # I_instr is Current added into cell by man-made instrument
    # I_instr_t is an array with shape(N)
    I_instr_t = args[0](N, t) # units microAmps/cm^2

    # Cell parameters
    C = 1 # Membrane capacitance; microFarads/cm^2
    # Shifted Nernst equilibrium potentials
    E_k = -12 # mV
    E_Na = 120 # mV
    E_L = 10.6 # mV
    # Maximal conductances
    g_k = 36 # mS/cm^2
    g_Na = 120 # mS/cm^2
    g_L = 0.3 # mS/cm^2
    e = 2.718281828459 # Euler's number

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

    # Equations of motion for state variables (should be shape N)
    dVdt = - 1.0/C * (g_k*(n**4) * (V-E_k) + g_Na*(m**3)*h*(V-E_Na) + g_L*(V-E_L) - I_instr_t)
    dndt = (n_inf - n)/tau_n
    dmdt = (m_inf - m)/tau_m
    dhdt = (h_inf - h)/tau_h

    # Has shape (4, N)
    dStatedt = np.array([dVdt, dndt, dmdt, dhdt])

    return dStatedt.flatten()