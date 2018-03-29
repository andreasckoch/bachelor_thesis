# plotte 2D plot (Zeit-Energy) aus SGR1806 Daten mit Photon Counts als Farbverlauf

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from utilities import get_data
import numpy as np


def plot_data(data, time_bins, energy_bins, title='', Norm=None, vmax=20):

    if Norm == 'Log':
        Norm = colors.LogNorm(vmin=1, vmax=np.max(data))
    else:
        Norm = None

    plt.imshow(data.T, cmap='inferno', norm=Norm, vmax=vmax, origin='lower', extent=[
               time_bins[0], time_bins[-1], energy_bins[0], energy_bins[-1]])
    plt.title(title)
    plt.ylabel('Energy Channels')
    plt.xlabel('Time in s')


if __name__ == "__main__":

    start_time = 845
    end_time = 1200
    time_pix = 2**12
    Norm = None  # 'Log'
    seperate_instruments = False

    binned_data, time_bins, energy_bins = get_data(
        start_time, end_time, time_pix, seperate_instruments=seperate_instruments, return_bins=True)

    if seperate_instruments:
        """
        plt.subplot(221)
        plot_data(np.sum(binned_data, axis=0), time_bins[0],
                  energy_bins[0], 'All Instruments summed', Norm)
        plt.subplot(222)
        plot_data(binned_data[0], time_bins[0], energy_bins[0], 'Instrument PCU0', Norm)
        plt.subplot(223)
        plot_data(binned_data[1], time_bins[1], energy_bins[1], 'Instrument PCU2', Norm)
        plt.subplot(224)
        plot_data(binned_data[2], time_bins[2], energy_bins[2], 'Instrument PCU3', Norm)
        """
        plt.subplots_adjust(wspace=0, hspace=0.2, top=0.96, bottom=0.06)
        # uncomment to only plot specific instrument
        #i=0;plot_data(binned_data[i], time_bins[i], energy_bins[i], 'Instrument PCU0', Norm)
        plot_data(np.sum(binned_data, axis=0), time_bins[0],
                  energy_bins[0], 'All Instruments summed', Norm)
    else:
        plot_data(binned_data, time_bins, energy_bins, 'All Instruments summed', Norm, vmax=30)
        plt.ylabel('Energy in keV')
        plt.subplots_adjust(left=0.04, right=0.98)

    plt.show()
    # plt.tight_layout()
    #plt.savefig("SGR1806_t_E_histogram_summed.png", dpi=1200)
    print("Done.")
