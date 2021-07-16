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


def process_spike_results(network_result_last_firing_array,
                          bogus_spike_time, N):
    spike_list = [[] for i in range(N)]  # Need N separate empty lists to record spike times to
    for n in range(N):
        sorted_unique_last_firing_of_neuron_n = np.unique(network_result_last_firing_array[:, n])
        for a_firing_time in sorted_unique_last_firing_of_neuron_n:
            if a_firing_time not in spike_list[n]:
                if a_firing_time != bogus_spike_time:
                    spike_list[n].append(a_firing_time)
    return spike_list