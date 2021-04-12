import numpy as np

def spike_list_to_array(N, spike_list):
    # Returns spike list in array form
    most_spikes = 0
    for n in range(N):
        for time_index in range(len(spike_list[n])):
            if time_index > most_spikes:
                most_spikes = time_index
    spike_list_array = np.zeros((N, most_spikes + 1))
    for n in range(N):
        for time_index in range(len(spike_list[n])):
            spike_list_array[n, time_index] = spike_list[n][time_index]
    return spike_list_array