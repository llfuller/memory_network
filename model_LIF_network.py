import numpy as np
import matplotlib.pyplot as plt
# def LIF_network(V_initial, times_array, *args):
def LIF_network(state_initial, times_array,
                dt, N, I_instr_t, R, C, threshold,
                last_firing_times, V_reset, refractory_time, g_syn_max,
                tau_syn, spike_list, use_STDP):
    # dt = args[0]
    # N = args[1]
    # I_instr_t = args[2]
    # R = args[3]
    # C = args[4]
    # threshold = args[5]
    # last_firing_times = args[6]
    # V_reset = args[7] # baseline to reset to after spiking
    # refractory_time = args[8]
    # print(state_initial)
    alpha = dt / C
    beta = dt/(C*R)
    V_t = np.zeros((times_array.shape[0], N))
    V_t[0,:] = state_initial[:N]
    W_t = np.zeros((times_array.shape[0], N, N))
    W_t[0,:,:] = state_initial[N : N+N*N].reshape((N,N))
    # print(W_t[0,:,:])
    last_firing_array = np.nan*np.ones((times_array.shape[0], N)) # nan allows easier raster plotting later
    # g_syn = g_syn_max * np.exp(-(t - last_firing_times) / tau_syn)
    for time_index, t in enumerate(times_array[:-1]):
        # Leaky integrate
        V_t[time_index+1,:] = (1-beta)*V_t[time_index,:]
        V_t[time_index+1,:] += alpha * I_instr_t(N,t)
        # a = a*((1-beta) + alpha * I_instr_t(N,t))
        # print(a)
        # print(I_instr_t(N,t))
        # print(V_t[time_index+1,:])
        # print("ALPHA:"+str(alpha))
        # Reset all neurons still in refractory to baseline
        in_refractory = ((t - last_firing_times) < refractory_time)
        (V_t[time_index+1,:])[in_refractory] = V_reset
        # Find V above threshold
        V_t_above_thresh = V_t[time_index, :] > threshold
        # Fire
        last_firing_times[V_t_above_thresh] = t
        # Synaptic currents due to firing
        # print(W_t[time_index]*g_syn_max)
        # print(np.shape(V_t_above_thresh))
        # print(np.tile(V_t_above_thresh,(N,1)))
        # print("shape of V_t: "+str(np.shape(V_t)))
        # print(W_t[time_index])
        V_t[time_index + 1, :] += alpha * g_syn_max* np.matmul(np.multiply(W_t[time_index],
                                                                           np.tile(V_t_above_thresh,(N,1)).transpose()),
                                                               V_t[time_index])
        # print(np.multiply(W_t[time_index],np.tile(V_t_above_thresh,(N,1)).transpose()))
        # if np.count_nonzero(np.multiply(W_t[time_index], np.tile(V_t_above_thresh,(N,1)).transpose())) != 0:
        #     print(np.multiply(W_t[time_index], np.tile(V_t_above_thresh,(N,1)).transpose()))
        # Reset to baseline if above threshold
        print(alpha * g_syn_max* np.matmul(np.multiply(W_t[time_index],np.tile(V_t_above_thresh,(N,1)).transpose()),
                                           V_t[time_index]))
        last_firing_array[time_index, :] = last_firing_times
        V_t[time_index+1, :][V_t_above_thresh] = V_reset
        if use_STDP:
            W_t[time_index+1, :, :] = W_t[time_index, :, :]-1000
    # plt.figure()
    # plt.plot(V_t)
    # plt.show()
    return [V_t, W_t, last_firing_array]



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