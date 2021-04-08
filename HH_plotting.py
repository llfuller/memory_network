import matplotlib.pyplot as plt

def plot_HH_single(sol):
    plt.figure()
    plt.plot(sol[:, 0])

    plt.figure()
    plt.plot(sol[:, 1:])
    plt.show()