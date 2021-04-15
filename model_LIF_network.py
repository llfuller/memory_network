import numpy as np
import matplotlib.pyplot as plt
# def LIF_network(V_initial, times_array, *args):
def LIF_network(state_initial, times_array,
                dt, N, I_instr_t, R, C, threshold,
                last_firing_times, V_reset, refractory_time, g_syn_max,
                E_syn, tau_syn, spike_list, use_STDP, STDP_scaling, tau_W, synapse_delay_delta_t):
    alpha = dt / C
    beta = dt/(C*R)
    V_t = np.zeros((times_array.shape[0], N))
    V_t[0,:] = state_initial[:N]
    last_firing_array = np.nan*np.ones((times_array.shape[0], N)) # nan allows easier raster plotting later
    W = state_initial[N : N+N*N].reshape((N,N))
    timesteps_synapse_delay = int(synapse_delay_delta_t/dt)
    print(timesteps_synapse_delay)
    def g_syn(g_syn_max, last_firing_times, tau_syn, t, synapse_delay_delta_t):
        t_after_firing = t - (last_firing_times + synapse_delay_delta_t)
        synapse_effect_on = t_after_firing.copy()
        synapse_effect_on[t_after_firing<0] = 0
        synapse_effect_on[t_after_firing>0] = 1

        return g_syn_max * np.multiply(np.exp(-t_after_firing / tau_syn),synapse_effect_on)

    for time_index, t in enumerate(times_array[:-1]):
        # Leaky integrate
        V_t[time_index+1,:] = (1-beta)*V_t[time_index,:]
        V_t[time_index+1,:] += alpha * I_instr_t(N,t)
        # Reset all neurons still in refractory to baseline
        in_refractory = ((t - last_firing_times) < refractory_time)
        (V_t[time_index+1,:])[in_refractory] = V_reset
        # Find V above threshold
        V_t_above_thresh = V_t[time_index, :] > threshold
        # Record firings
        last_firing_times[V_t_above_thresh] = t
        # Add synaptic currents due to firing
        V_t[time_index + 1, :] += alpha * g_syn_max* \
                                  np.multiply(np.matmul(W,
                                                        g_syn(g_syn_max, last_firing_times, tau_syn, t,
                                                              synapse_delay_delta_t)),
                                              (E_syn-V_t[time_index-timesteps_synapse_delay]))


        # Reset to baseline if above threshold
        last_firing_array[time_index, :] = last_firing_times
        V_t[time_index+1, :][V_t_above_thresh] = V_reset
        # Reset to baseline if below baseline:
        # Find V below baseline
        V_t_below_reset = V_t[time_index, :] < V_reset
        V_t[time_index + 1, :][V_t_below_reset] = V_reset
        # Adapt weights into postsynaptic neurons (index i) that have just fired according to V_t_above_thresh
        if use_STDP:
            timing_difference = np.subtract.outer(last_firing_times, last_firing_times)  # positive if pre spiked first
            delta_W = STDP_scaling \
                           * np.multiply(np.sign(timing_difference), np.exp(-np.abs(timing_difference)/tau_W))
            delta_W[W==0] = 0 # Make sure cells without connections do not develop them
            delta_W[delta_W<0] = 0 # Ensure conductances don't drop below zero (which would switch excite/inhibit)
            W += delta_W
    print(W.shape)
    return [V_t, W, last_firing_array]



def LIF_network2(state, t, *args):
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