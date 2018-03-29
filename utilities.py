import numpy as np

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
    if energy_pix.size == 1:
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


def get_data(start_time, end_time, time_pix, seperate_instruments=False, return_bins=False):
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
    data_path = "/home/marvin/data/SGR1806/SGR1806_time_PCUID_energychannel.txt"
    data = np.loadtxt(data_path).transpose()
    data[0] = data[0] - data[0].min()
    data = data[:, np.argmax(data[0] > float(start_time)):np.argmax(data[0] > float(end_time))]

    # convert channels to energy in keV
    if not seperate_instruments:
        energy_path = "/home/marvin/data/SGR1806/energy_channels.txt"
        energy = np.loadtxt(energy_path, usecols=[6, 7], skiprows=25).transpose()
        instrument = np.array(data[1], dtype=int)
        instrument[instrument > 0] = 1  # distinguish between PCU0:=0 and PCU1234:=1 energy
        channel = np.array(data[2], dtype=int)
        data = np.array([data[0], energy[instrument, channel]])

    # bin data, create histogram
    if seperate_instruments:
        energy_pix = int(np.max(data[2])+1)
        binned_data, time_bins, energy_bins = create_histograms(data, time_pix, energy_pix)
    else:
        energy_pix = 256
        binned_data, time_bins, energy_bins = np.histogram2d(
            data[0], data[1], bins=[time_pix, energy_pix])

    if return_bins:
        return binned_data, time_bins, energy_bins
    else:
        return binned_data
