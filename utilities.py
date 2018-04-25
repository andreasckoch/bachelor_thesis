import numpy as np
import nifty4 as ift
import scipy.sparse
import scipy.sparse.linalg
import constants as const
import matplotlib.pyplot as plt

data_path = const.data_path
energy_path = const.energy_path


def get_time_mask(data, domain, threshold=2):

    # When extracting real mask, consider start and end point, because it will be applied after signal is padded
    # if binned_data is given, sum along instrument axis=0, and along energy axis=1
    if data.shape[0] == 3:
        data = np.sum(np.sum(data, axis=0), axis=1)
    else:
        data = np.sum(data, axis=1)

    data_mask = np.ones(domain.shape)
    NotData = False
    dead_count = 0

    for i in range(data.shape[0]):
        if i == data.shape[0] - (threshold - 1):
            break
        if np.sum(data[i:i + threshold]) == 0:
            data_mask[i:i + threshold] = 1e-10
            if NotData is False:
                dead_count += 1
            NotData = True
        else:
            NotData = False

    print('Detected %d dead intervalls in the data.' % (int(dead_count)))
    #np.savetxt('/home/marvin/code/Marvin_Baumann/data_mask-1.txt', data_mask)

    return ift.Field(domain, val=data_mask)
    # data_mask wird korrekt erkannt (vier Totstellen für time_pix=2**12 und 845-1200s)
    # (drei Totstellen für time_pix=2**12 und 845-1300s)

    # maske muss für die 2d response ein ift feld mit signal domain nach dem time padding sein!


def create_histograms(data, time_pix):
    # create histograms seperately for each instrument and save in 3D array binned_data

    # problem beim binning entlang der Energie, denn PCU23 haben keine Counts in channel 0,1
    # welche manuell eingefügt werden. Dementsprechend wird PCU0 mit 256 bins  und PCU23 mit
    # 254 bins in der Energie gebinnt. Danach werden die ersten beiden (leeren) bins zu PCU23
    # hinzugefügt.

    binned_data = np.zeros(shape=(3, time_pix, 256), dtype=np.float64)
    time_bins = np.zeros(shape=(3, time_pix+1))
    energy_bins = np.zeros(shape=(3, 257))

    binned_data[0], time_bins[0], energy_bins[0] = np.histogram2d(
        data[0, data[1] == 0], data[2, data[1] == 0], bins=[time_pix, 256])

    binned_data[1, :, 2:], time_bins[1], energy_bins[1, 2:] = np.histogram2d(
        data[0, data[1] == 2], data[2, data[1] == 2], bins=[time_pix, 254])

    binned_data[2, :, 2:], time_bins[2], energy_bins[2, 2:] = np.histogram2d(
        data[0, data[1] == 3], data[2, data[1] == 3], bins=[time_pix, 254])

    # add zeros for energy channels 0 and 1, because they are not caught by binning
    # binned_data[1, :, :2] = 1e-3
    # binned_data[2, :, :2] = 1e-3
    # binned_data[2] = np.insert(binned_data[2], 0, np.zeros(shape=(time_pix, 2)))
    # binned_data[1] = np.insert(binned_data[1], 0, np.zeros(shape=(time_pix, 2)))

    # add appropriate energy_bins, for plotting to work later on
    energy_bins[1, 1] = 1
    energy_bins[2, 1] = 1

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
    data = np.loadtxt(data_path).transpose()
    data[0] = data[0] - data[0].min()
    von = np.argmax(data[0] > float(start_time))
    bis = np.argmax(data[0] > float(end_time))
    data = data[:, von:bis]

    # convert channels to energy in keV
    if not seperate_instruments:
        energy = np.loadtxt(energy_path, usecols=[6, 7], skiprows=25).transpose()
        instrument = np.array(data[1], dtype=int)
        instrument[instrument > 0] = 1  # distinguish between PCU0:=0 and PCU1234:=1 energy
        channel = np.array(data[2], dtype=int)
        data = np.array([data[0], energy[instrument, channel]])

    # bin data, create histogram
    if seperate_instruments:
        binned_data, time_bins, energy_bins = create_histograms(data, time_pix)
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


def get_instrument_factors(length=1):
    # create a response from all data available
    data = np.loadtxt(data_path, usecols=[1, 2]).transpose()
    # as for instruments 2 and 3 no photons are registered in the first two bins,
    # histogram does not those bins up --> insert 0s
    out = np.array([np.histogram(data[1, data[0] == 0], bins=256)[0].astype(float),
                    np.insert(np.histogram(data[1, data[0] == 2], bins=254)[
                        0].astype(float), 0, [1e-3, 1e-3]),
                    np.insert(np.histogram(data[1, data[0] == 3], bins=254)[0].astype(float), 0, [1e-3, 1e-3])])
    if length == 1:
        x = np.zeros((1,))
    else:
        x = np.linspace(-20, 20, length)

    x = kernel(x, 0.01)
    padd = np.zeros((out.shape[0], out.shape[1] + length))
    padd[:, length // 2:-length//2] = out
    for i in range(out.shape[1] - length):
        out[:, i] = np.sum(padd[:, i:i + length] * x[np.newaxis, :], axis=1) / length

    return out


def kernel(x, sigma):
    tmp = x*x
    tmp *= -2.*np.pi*np.pi*sigma*sigma
    np.exp(tmp, out=tmp)
    return tmp


def scale_and_normalize(x, instrument_factors):
    # Faktoren aus Photon Count Daten pro Channel (instrument_factors) skalieren und
    # mittels sinnvoller Normierung (dim(a) / sum(a) mit a = instrument_factors) multiplizieren.
    # Sinnvolle Normierung, da Mittelwert der gesamten Faktoren a_mittel = 1/dim(a) * sum(dim(a) * a / sum(a)) = 1
    # Input/Output Dimensions: 3 x t_pix x 256
    if isinstance(x, ift.Field):
        x = x.val
    f = x.shape[2] * instrument_factors[:, np.newaxis, :] / np.sum(instrument_factors, axis=1)[:, np.newaxis, np.newaxis]
    print(np.amin(f))
    x = x.copy() * f
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


def build_energy_response(signal_domain):

    # Load constants
    dE = signal_domain[1].distances[0]
    t_pix = signal_domain.shape[0]//2
    e_pix = signal_domain.shape[1]
    energy_dicts, energies = get_dicts(True, True)
    ins_p = const.ins_p

    # f: fractions mit denen Eintrag aus Signal multipliziert wird
    # s: 1-D Indizes der Signalelemente
    # d: 1-D Indizes der Datenelemente
    f = []
    s = []
    d = []

    for ins in [0, 1, 2]:
        s_i = 0  # Index des zuletzt geschnittenen signal bins
        p0 = 1.0  # verbleibender Bruchteil des zuletzt geschnittenen signal bins
        for e in energies[ins]:
            """
            i bis j sind signal bins (entlang Energie), die auf diesen data energy bin e aufgeteilt werden.

            Kreiere Listelemente (fraction ist Anteil für jeweiligen Channel):
            für i:  f0,0 = ins_p* p0*fraction0  für ersten Channel dieses data energy bins und erster time pixel
                    ...
                    f0,t_pix = ins_p* p0*fraction0 # selbiger wie f0,0 aber für letzten time pixel
            Indizes:   
                    # Signal-Indizes mit Energie pixel index i und allen time_pixels
                    s0[0:t_pix] = np.ravel_multi_index([range(t_pix), [i,]], (t_pix, e_pix))
                    # Daten-Indizes des Instruments ins und Channels channel0 mit allen time_pixels
                    d0[0:t_pix] = np.ravel_multi_index([[ins,],range(t_pix), [channel0,]], (3,t_pix, 256))

                    f1,0 = ins_p* p0*fraction1  für zweiten Channel...
                    ...
                    f2,0 = ins_p* p0*fraction2  für dritten
                    ...
                    f3,0 = ins_p* p0*fraction3  für vierten
                    ...
            für     f4,0 = ins_p* fraction0     für ersten Channel dieses data energy bins
            i+1:j : ...
                    f5,0 = ins_p* fraction1     ...
                    ...
                    f6,0 = ins_p* fraction2
                    ...
                    f7,0 = ins_p* fraction3
                    ...
            für j:  f8,0 = ins_p* (1-p1)*fraction0
                    ...
                    f9,0 = ins_p* (1-p1)*fraction1
                    ...
                    f10,0 = ins_p* (1-p1)*fraction2
                    ...
                    f11,0 = ins_p* (1-p1)*fraction3
                    ...
                    f11,t_pix = ins_p* (1-p1)*fraction3

            """

            # finde signal bin index, der von e geschnitten wird
            s_j = int(e//dE)
            p1 = ((s_j+1)*dE-e)/dE  # neuer verbleibender Bruchteil des signal bins s_j

            if s_i == s_j:
                for frac_i, channel in enumerate(energy_dicts[ins][e][0]):
                    f_tmp = [ins_p[ins]*(p0-p1)*energy_dicts[ins][e][1][frac_i]]*t_pix
                    s_tmp = np.ravel_multi_index([range(t_pix), [s_i, ]], (t_pix, e_pix))
                    d_tmp = np.ravel_multi_index([[ins, ], range(t_pix), [channel, ]], (3, t_pix, 256))
                    f.extend(f_tmp)
                    s.extend(s_tmp)
                    d.extend(d_tmp)
            else:
                for frac_i, channel in enumerate(energy_dicts[ins][e][0]):

                    # s_i
                    f_tmp = [ins_p[ins]*p0*energy_dicts[ins][e][1][frac_i]]*t_pix
                    s_tmp = np.ravel_multi_index([range(t_pix), [s_i, ]], (t_pix, e_pix))
                    d_tmp = np.ravel_multi_index([[ins, ], range(t_pix), [channel, ]], (3, t_pix, 256))
                    f.extend(f_tmp)
                    s.extend(s_tmp)
                    d.extend(d_tmp)

                    # s_i+1:s_j
                    for s_ij in range(s_i+1, s_j):
                        f_tmp = [ins_p[ins]*energy_dicts[ins][e][1][frac_i]]*t_pix
                        s_tmp = np.ravel_multi_index([range(t_pix), [s_ij, ]], (t_pix, e_pix))
                        d_tmp = np.ravel_multi_index([[ins, ], range(t_pix), [channel, ]], (3, t_pix, 256))
                        f.extend(f_tmp)
                        s.extend(s_tmp)
                        d.extend(d_tmp)

                    # s_j
                    if s_j < e_pix:
                        f_tmp = [ins_p[ins]*(1-p1)*energy_dicts[ins][e][1][frac_i]]*t_pix
                        s_tmp = np.ravel_multi_index([range(t_pix), [s_j, ]], (t_pix, e_pix))
                        d_tmp = np.ravel_multi_index([[ins, ], range(t_pix), [channel, ]], (3, t_pix, 256))
                        f.extend(f_tmp)
                        s.extend(s_tmp)
                        d.extend(d_tmp)

            # setze Werte für nächsten Energie bin
            p0 = p1
            s_i = s_j

    energy_response = scipy.sparse.coo_matrix((f, (d, s)), shape=(3*t_pix*256, t_pix*e_pix)).tocsc()
    return scipy.sparse.linalg.aslinearoperator(energy_response)


def get_dicts(return_energies=False, return_channel_fractions=False):
    """
    return_energies == True : return array of energy bins per instrument

    return_channel_fractions==False : return energy_dicts without fractions, with PCU2 and PCU3 taken together

    if both true (usual usecase), energies looks like this:
        [[1.95 .. 126.87]    # energy levels of PCU0 (as numpy array)
         [2.06 .. 117.86]    # energy levels of PCU2 (as numpy array)
         [2.06 .. 117.86]]  # energy levels of PCU3 (as numpy array)

    energy_dicts looks like this:
    [  # dict for PCU0, which keys are the energy bins of PCU0
     { 1.95: [[0, 1, 2, 3, 4], # numbers of energy channels that feed into this energy bin
    [0.0008764241893076249,    # fraction of photons that go into this channel
     0.0008764241893076249,    # in case they fell into this energy bin 1.95keV
     0.007011393514460999,
     0.28659070990359337,
     0.7046450482033304]],
     ...
    },
    {...}, # dict for PCU2 with same structure as above
    {...}  # dict for PCU3 with same structure as above
    ]

    Example:
    To get Channel numbers of the first energy bin of the first intrument (which has energy 1.95keV), do this:
    instrument = 0 # first instrument
    energy_bin = energies[instrument][0] # read out first energy of instrument
    # get channel numbers of energy_bin
    print(energy_dicts[instrument][energy_bin][0]) # Result: [0, 1, 2, 3, 4]
    # get fractions of the channels of this energy bin (take second element)
    print(energy_dicts[instrument][energy_bin][1])
    # Result: [0.0008764241893076249, 0.0008764241893076249, 0.007011393514460999, 0.28659070990359337, 0.7046450482033304]]



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

        energy_dicts[1][2.06][1][0] = 1e-5
        energy_dicts[1][2.06][1][1] = 1e-5
        energy_dicts[2][2.06][1][0] = 1e-5
        energy_dicts[2][2.06][1][1] = 1e-5

    if return_energies:
        return energy_dicts, energies
    else:
        return energy_dicts


if __name__ == "__main__":
    factors = get_instrument_factors(length=3)
    x = np.linspace(1, 256, num=256)

    plt.subplot(221)
    plt.bar(x, height=factors[0], width=1)
    plt.title('Factors PCU0')
    plt.xlabel('Photon Count')
    plt.ylabel('Energy Channels]')

    plt.subplot(222)
    plt.bar(x, height=factors[1], width=1)
    plt.title('Factors PCU2')
    plt.xlabel('Photon Count')
    plt.ylabel('Energy Channels]')

    plt.subplot(223)
    plt.bar(x, height=factors[2], width=1)
    plt.title('Factors PCU3')
    plt.xlabel('Photon Count')
    plt.ylabel('Energy Channels]')

    plt.subplots_adjust(left=0.04, right=0.99, hspace=0.23, top=0.95, bottom=0.06)
    plt.show()
