import numpy as np
import nifty4 as ift

"""
sensible values:
start_time = 845
end_time = 1200
time_pix = 2**12
"""
data_path = "/home/andi/bachelor/data/originaldata/SGR1806_time_PCUID_energychannel.txt"
energy_path = "/home/andi/bachelor/data/arrangeddata/energy_channels.txt"
ins_p = [341909/(341909+335600+329606), 335600 /
         (341909+335600+329606), 329606/(341909+335600+329606)]


def get_time_mask(data, threshold=2):

    # When extracting real mask, consider start and end point, because it will be applied after signal is padded
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

    # maske muss für die 2d response ein ift feld mit signal domain nach dem time padding sein!


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


def get_instrument_factors():
    # create a response from all data available
    data = np.loadtxt(data_path, usecols=[1, 2]).transpose()
    # as for instruments 2 and 3 no photons are registered in the first two bins,
    # histogram does not those bins up --> insert 0s
    out = np.array([np.histogram(data[1, data[0] == 0], bins=256)[0].astype(float),
                    np.insert(np.histogram(data[1, data[0] == 2], bins=254)[
                        0].astype(float), 0, [1e-3, 1e-3]),
                    np.insert(np.histogram(data[1, data[0] == 3], bins=254)[0].astype(float), 0, [1e-3, 1e-3])])
    return out


def scale_and_normalize(x, instrument_factors):
    # Faktoren aus Photon Count Daten pro Channel (instrument_factors) multiplizieren und
    # mittels sinnvoller Skalierung (sum(x) / sum(a*x) mit a = instrument_factors) normieren
    # Input/Output Dimensions: 3 x t_pix x 256
    if isinstance(x, ift.Field):
        x = x.val
    x_0 = x.copy()
    x_sum_0 = np.sum(x_0, axis=2)

    x *= instrument_factors[:, np.newaxis, :]
    x_sum = np.sum(x, axis=2)

    # for numerical stability, change every 0 in x_sum to 1e-3.
    # This does not change the factor as x_sum_0 is also 0 for all changed components.
    # If D4PO does not see any 0s then take this part out! Because somehow all masked out parts don't have 0s anymore.
    for i in range(x_sum.shape[0]):
        for j in range(x_sum.shape[1]):
            if x_sum[i, j] == 0 and x_sum_0[i, j] == 0:
                x_sum[i, j] = 1e-3

    x = x * x_sum_0[:, :, np.newaxis] / x_sum[:, :, np.newaxis]

    return x


def get_energy_widths(energy_bins):
    # calculate width of the energy bin of each channel
    # energy_bins should have dim: 2 x 256
    energy_bins_width = np.zeros((3, 256))
    for i in range(energy_bins.shape[0]):
        for j in range(energy_bins.shape[1]):
            if j == 0:  # first width is difference of component to 0
                energy_bins_width[:2, 0] = energy_bins[:, 0]
            elif energy_bins[i, j] != energy_bins[i, j - 1]:  # here the calculation happens
                energy_bins_width[i, j] = energy_bins[i, j] - energy_bins[i, j - 1]
            else:
                energy_bins_width[i, j] = energy_bins_width[i, j - 1]  # if two channels have the same bin

    energy_bins_width[2, :] = energy_bins_width[1, :]

    return energy_bins_width


def get_mean_energy(energy_bins):
    # INPUT: N x number of instruments dimensionality, where N refers to number of channels
    # Only 1 Channel per Component allowed
    energy_bins_mean = energy_bins.copy()

    for i in range(energy_bins.shape[0]):
        if i is not 0:
            # consider all instrument rows
            energy_bins_mean[i, :] = (energy_bins[i, :] + energy_bins[i - 1, :]) / 2

    return energy_bins_mean


def energy_response(s, energy_dicts=None, energies=None):
    """
    Take in signal vector s, which has elements of photon counts in energy bins in the signal domain.
    s is of type ift.Field. Its first component gives the photon count in the energy intervall:
    [0 keV, s.domain.distances kev]

    Output: data array with shape (3,256)
    """
    # energy bin width in signal
    dE = s.domain[1].distances[0]
    if energy_dicts is None or energies is None:
        energy_dicts, energies = get_dicts(
            return_energies=True, return_channel_fractions=True)

    # 1. Aufteilung auf Instrumente, signal dim: 3 x t_pix x e_pix
    signal = np.array([s.val*ins_p[0], s.val*ins_p[1], s.val*ins_p[2]])
    data = np.zeros(shape=(3, s.shape[0], 256))

    # 2. Einordnung in Energie bins der Instrumente
    i = 0
    p0 = 1.0  # verbleibender Bruchteil des aktuellen signal bins
    for ins in [0, 1, 2]:
        for e in energies[ins]:
            # finde signal bin, der von e geschnitten wird
            j = int(e//dE)
            p1 = ((j+1)*dE-e)/dE  # neuer verbleibender Bruchteil des signal bins j

            if i == j:
                photons_in_e_bin = (p0-p1)*signal[ins, :, i]
            else:
                # Restanteil von bin i + alle bins zwischen i und j + Anteil von bin j
                photons_in_e_bin = p0*signal[ins, :, i] + \
                    np.sum(signal[ins, :, i+1:j], axis=1) + (1-p1)*signal[ins, :, j]

            # 3. Aufteilung auf Channels
            data[ins, :, energy_dicts[ins][e][0]] = np.ma.outerproduct(photons_in_e_bin,
                                                                       np.array([energy_dicts[ins][e][1]])).transpose()

            # setze Werte für nächste Energie
            p0 = p1
            i = j

    return data


def energy_response_adjoint(data, domain, energy_dicts=None, energies=None):
    """
    Input: data as ift.Field(domain=UnstructuredDomain) with shape (3, tpix, 256)
           domain of signal field
    Output: numpy array of signal field with photon counts in energy bins as elements
    """

    # energy bin width in signal
    dE = domain[1].distances[0]
    signal = np.zeros(shape=domain.shape)
    # take first element to be photon counts in energy intervall [0 keV, dE keV]

    for ins in [0, 1, 2]:
        i = 0  # letzter geschnittener Signal bin
        p0 = 1.0  # verbleibender Bruchteil des signal bins i
        for e in energies[ins]:
            # Sammle alle Photon Counts zusammen, für diesen energy bin des instrument ins
            # energy_dicts[ins][e][0] ist Liste an Channel numbers, die e befüllen, für instrument ins
            photons_in_e_bin = np.sum(data[ins, :, energy_dicts[ins][e][0]].T, axis=1)  # Wieso transpose???

            # Teile Photon Counts dieses energy bins auf die signal bins auf, die davon geschnitten werden
            # (unteren und oberen Schnittpunkt und die Bruchteile finden, mit denen diese zu füllen sind)
            j = int(e//dE)  # oberer geschnittener signal bin
            p1 = ((j+1)*dE-e)/dE  # neuer verbleibender Bruchteil des signal bins j
            norm = p0+(1-p1)+(j-1-i)
            # Teile Photon Counts den Signal bins zu, die von bin e geschnitten werden
            signal[:, i] += p0*photons_in_e_bin/norm
            signal[:, i+1:j] += photons_in_e_bin[:, np.newaxis]/norm
            signal[:, j] += (1-p1)*photons_in_e_bin/norm

            i = j
            p0 = p1

    return signal


def get_dicts(return_energies=False, return_channel_fractions=False):
    """
    return_energies == True : return array of energy bins per instrument

    return_channel_fractions==False : return energy_dicts without fractions, with PCU2 and PCU3 taken together

    """

    energies = np.loadtxt(energy_path, usecols=[6, 7], skiprows=25).transpose()
    unique, counts = np.unique(energies[0], return_counts=True)
    energies_PCU0 = dict(zip(unique, counts))
    for e in unique:
        energies_PCU0[e] = [np.argmax(energies[0] >= e)+n for n in range(energies_PCU0[e])]
    unique, counts = np.unique(energies[1], return_counts=True)
    energies_PCU23 = dict(zip(unique, counts))
    for e in unique:
        energies_PCU23[e] = [np.argmax(energies[1] >= e)+n for n in range(energies_PCU23[e])]

    energy_dicts = [energies_PCU0, energies_PCU23]
    # print(energy_dicts[0][104.46])

    if return_channel_fractions:
        data = np.loadtxt(data_path, usecols=[1, 2]).transpose()
        d = []

        for i in [0, 2, 3]:
            unique, counts = np.unique(data[1, data[0] == i], return_counts=True)
            d.append(dict(zip(unique.astype(int), counts)))

        # Add these into dictionary
        d[1][0] = 0
        d[1][1] = 0
        d[2][0] = 0
        d[2][1] = 0
        energies = [np.unique(energies[0]), np.unique(energies[1]), np.unique(energies[1]).copy()]
        energy_dicts = [energy_dicts[0], energy_dicts[1], energy_dicts[1].copy()]

        for i in [0, 1, 2]:
            for e in energies[i]:
                norm = np.sum(np.array([d[i][n] for n in energy_dicts[i][e]]))
                p = [d[i][n]/norm for n in energy_dicts[i][e]]
                energy_dicts[i][e] = [energy_dicts[i][e], p]

    if return_energies:
        return energy_dicts, energies
    else:
        return energy_dicts


if __name__ == "__main__":
    get_instrument_response()
