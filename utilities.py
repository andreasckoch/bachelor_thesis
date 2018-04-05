import numpy as np
import nifty4 as ift
import copy

"""
sensible values:
start_time = 845
end_time = 1200
time_pix = 2**12
"""


def get_time_mask(data, threshold=2):

    # if binned_data is given, sum along instrument axis=0, and along energy axis=1
    data = np.sum(np.sum(data, axis=0), axis=1)

    data_mask = np.ones_like(data)
    NotData = False
    dead_count = 0

    for i in range(data.shape[0]):
        if i == data.shape[0] - (threshold - 1):
            break
        if np.sum(data[i:i + threshold]) == 0:
            data_mask[i:i + threshold] = 0
            if NotData is False:
                dead_count += 1
            NotData = True
        else:
            NotData = False

    print('Detected %d dead intervalls in the data.' % (int(dead_count)))
    #np.savetxt('/home/marvin/code/Marvin_Baumann/data_mask-1.txt', data_mask)

    return data_mask
    # data_mask wird korrekt erkannt (vier Totstellen für time_pix=2**12 und 845-1200s)
    # (drei Totstellen für time_pix=2**12 und 845-1300s)


def create_histograms(data, time_pix, energy_pix):
    # create histograms seperately for each instrument and save in 3D array binned_data

    binned_data = np.zeros(shape=(3, time_pix, energy_pix))
    time_bins = np.zeros(shape=(3, time_pix+1))
    if energy_pix.size == 1:   # was genau wird hier gmeacht??
        energy_bins = np.zeros(shape=(3, energy_pix+1))
    else:
        energy_bins = energy_pix  # in case i give the bins directly

    binned_data[0], time_bins[0], energy_bins[0] = np.histogram2d(
        data[0, data[1] == 0], data[2, data[1] == 0], bins=[time_pix, energy_pix])

    binned_data[1], time_bins[1], energy_bins[1] = np.histogram2d(
        data[0, data[1] == 2], data[2, data[1] == 2], bins=[time_pix, energy_pix])

    binned_data[2], time_bins[2], energy_bins[2] = np.histogram2d(
        data[0, data[1] == 3], data[2, data[1] == 3], bins=[time_pix, energy_pix])

    return binned_data, time_bins, energy_bins


def get_data(start_time, end_time, time_pix, seperate_instruments=False, return_bins=False, return_all_data=False):
    """
    Helper function to read in SGR1806 data and return the binned data (i.e. a histogram)

    seperate_instruments==True:
        seperate data by instrument
        use channel number as energy data (conversion to keV has to be done in Response R)
    seperate_instruments==False:
        sum the events of all instruments
        convert channel number to energy levels ins keV

    """

    # load data and select time intervall
    data_path = "/home/andi/bachelor/data/originaldata/SGR1806_time_PCUID_energychannel.txt"
    data = np.loadtxt(data_path).transpose()
    data[0] = data[0] - data[0].min()

    if return_all_data:
        return data

    # if start and end time were specified
    data = data[:, np.argmax(data[0] > float(start_time)):np.argmax(data[0] > float(end_time))]

    # convert channels to energy in keV
    if not seperate_instruments:
        energy_path = "/home/andi/bachelor/data/arrangeddata/energy_channels.txt"
        energy = np.loadtxt(energy_path, usecols=[
                            6, 7], skiprows=25).transpose()
        instrument = np.array(data[1], dtype=int)
        # distinguish between PCU0:=0 and PCU1234:=1 energy
        instrument[instrument > 0] = 1
        channel = np.array(data[2], dtype=int)
        data = np.array([data[0], energy[instrument, channel]])

    # bin data, create histogram
    if seperate_instruments:
        energy_pix = int(np.max(data[2])+1)
        binned_data, time_bins, energy_bins = create_histograms(
            data, time_pix, energy_pix)
    else:
        energy_pix = 256
        binned_data, time_bins, energy_bins = np.histogram2d(
            data[0], data[1], bins=[time_pix, energy_pix])

    if return_bins:
        return binned_data, time_bins, energy_bins
    else:
        return binned_data


#        -Binned_data: 3 histograms generated with np.histogram2d using time_pix (f.e. 2**12) and energy_pix (channel amount)
#        -time_bins: 3 vectors signaling at which timestamps a new bin starts
#        -energy_bins: same as with time, but in contrast to time bins, they are not uniform!
#        -wanted_energy_bins: desired uniform energy bins, dead or alive


def channel_calibration(data):
    print(data[2])
    unique, counts = np.unique(data[2], return_counts=True)
    d = dict(zip(unique, counts))
    return d


def effectve_area():
    # OUTPUT: N x number of instruments dimensionality, first row gives

    energy_path = "/home/andi/bachelor/data/arrangeddata/energy_channels.txt"
    effective_area_path = "/home/andi/bachelor/data/originaldata/EffectiveArea.txt"
    energy_bins = np.loadtxt(energy_path, usecols=[6, 7], skiprows=25)
    energy_bins_mean = get_mean_energy(energy_bins)
    effectve_area_energy = np.loadtxt(effective_area_path, delimiter=", ")
    effectve_area = copy.copy(energy_bins_mean)

    # find nearest energy point to mean energies for its effective area
    indices_1 = np.searchsorted(effectve_area_energy[:, 0], energy_bins_mean[:, 0])
    indices_2 = np.searchsorted(effectve_area_energy[:, 0], energy_bins_mean[:, 1])
    effectve_area[:, 0] = effectve_area_energy[indices_1, 1]
    effectve_area[:, 1] = effectve_area_energy[indices_2, 1]

    print(effectve_area)
    return effectve_area


def get_mean_energy(energy_bins):
    # INPUT: N x (1 + number of instruments) dimensionality, where N refers to number of channels
    # Only 1 Channel per Component allowed
    energy_bins_mean = copy.copy(energy_bins)

    for i in range(energy_bins.shape[0]):
        if i is not 0:
            # consider all instrument rows
            energy_bins_mean[i, :] = (energy_bins[i, :] + energy_bins[i - 1, :]) / 2

    return energy_bins_mean


if __name__ == "__main__":
    effectve_area()
