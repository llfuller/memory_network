import matplotlib.pyplot as plt

def plot_single_neuron(sol):

    # Plot voltages
    plt.figure()
    plt.plot(sol[:, 0])
    plt.ylabel('Voltage (mV)')
    plt.xlabel('Time (ms)')

    # Plot gating variables
    plt.figure()
    plt.plot(sol[:, 1:])
    plt.ylabel('Gating Variables')
    plt.xlabel('Time (ms)')

    plt.show()


def plot_many_neurons(N, times_array, sol):
    # sol has shape (timesteps, 4, N)

    for i in range(N):
        print("Plotting for elements " + str(0+i*4) + " to " + str(4+i*4))

        # Plot voltages
        plt.figure()
        plt.plot(times_array, sol[:, 0, i])
        plt.ylabel('Voltage (mV)')
        plt.xlabel('Time (ms)')

        # Plot gating variables
        plt.figure()
        plt.plot(times_array, sol[:, 1:4, i])
        plt.ylabel('Gating Variables')
        plt.xlabel('Time (ms)')
        plt.show()

def plot_many_neurons_simultaneous(N, times_array, sol, use_STDP, extra_descriptors=''):
    # sol has shape (timesteps, 4 or 2, N)
    plt.figure()

    for i in range(N):
        # Plot voltages
        plt.plot(times_array, sol[:, 0, i])
        plt.ylabel('Voltage (mV)')
        plt.xlabel('Time (ms)')
        if use_STDP == True:
            plt.title('Voltages with STDP')
        else:
            plt.title('Voltages without STDP')
    plt.savefig('plots/V;STDP='+ str(use_STDP) +';'+ extra_descriptors)

def make_raster_plot(N, spike_list, use_STDP, extra_descriptors=''):
    # Makes raster plot of neuron spikes
    number_list = []
    spike_list_flattened = []
    for n in range(len(spike_list)):
        for a_time in spike_list[n]:
            number_list.append(n)
            spike_list_flattened.append(a_time)

    plt.figure()
    plt.scatter(spike_list_flattened, number_list)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Label')
    if use_STDP == True:
        plt.title('Spike Timings with STDP')
    else:
        plt.title('Spike Timings without STDP')
    plt.grid(True, axis='y')
    plt.savefig('plots/Raster;STDP='+ str(use_STDP) +';'+ extra_descriptors)