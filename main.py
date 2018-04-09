import QPO
import mock_signals
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    nbins = 2**10
    x = np.linspace(0, 130, num=nbins)
    channels = np.linspace(0, 255, num=256)
    x_ticks = np.linspace(0, 250, num=26)

    s = mock_signals.mock_signal_energy(nbins)
    R_E = QPO.EnergyResponse(s.domain)
    #time_mask = QPOutils.get_time_mask(data)
    #R_t = Response(time_mask)

    lam = R_E(s)
    data = lam.val
    print(np.sum(s.val), np.sum(data, axis=1))

    plt.subplot(221)
    plt.bar(x, height=s.val, width=130/nbins+0.05)
    plt.title('Signal')
    plt.xlabel('Energy in keV')
    plt.ylabel('Photon counts')
    plt.subplot(222)
    plt.bar(channels, height=data[0], width=1)
    plt.title('Data for PCU0')
    plt.xlabel('Channel number')
    plt.ylabel('Photon counts')
    plt.xticks(x_ticks)
    plt.grid(which='both')
    plt.subplot(223)
    plt.bar(channels, height=data[1], width=1)
    plt.title('Data for PCU2')
    plt.xlabel('Channel number')
    plt.ylabel('Photon counts')
    plt.xticks(x_ticks)
    plt.grid(which='both')
    plt.subplot(224)
    plt.bar(channels, height=data[2], width=1)
    plt.title('Data for PCU3')
    plt.xlabel('Channel number')
    plt.ylabel('Photon counts')
    plt.xticks(x_ticks)
    plt.grid(which='both')

    plt.subplots_adjust(left=0.04, right=0.99, hspace=0.23, top=0.95, bottom=0.06)
    plt.show()
