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


def plot_many_neurons(N, sol):
    # sol has shape (timesteps, 4, N)
    for i in range(N):
        print("Plotting for elements " + str(0+i*4) + " to " + str(4+i*4))

        # Plot voltages
        plt.figure()
        plt.plot(sol[:, 0, i])
        plt.ylabel('Voltage (mV)')
        plt.xlabel('Time (ms)')

        # Plot gating variables
        plt.figure()
        plt.plot(sol[:, 1:4, i])
        plt.ylabel('Gating Variables')
        plt.xlabel('Time (ms)')
        plt.show()
