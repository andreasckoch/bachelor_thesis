import numpy as np
import matplotlib.pyplot as plt
from plot_data import plot_data

energy_path = "/home/andi/bachelor/data/arrangeddata/energy_channels.txt"
energy_remapped_path = "/home/andi/bachelor/data/arrangeddata/energy_channels_remapped.txt"
data_path = "/home/andi/bachelor/data/originaldata/SGR1806_time_PCUID_energychannel.txt"
data = np.loadtxt(data_path).transpose()

# to fix problems, use explicit bins, i.e. the energy levels specified in energy_channels
# and add one additional value at the end to define the most righthand bin edge

energies = np.loadtxt(energy_path, usecols=[0, 1, 6, 7], skiprows=25).transpose()
remapped_energies = np.loadtxt(energy_remapped_path, usecols=[0, 1, 2]).transpose()


plt.subplot(431)
plt.hist(data[2, data[1] == 0], bins=256)
plt.title('Channels for PCU0')
plt.ylabel('Photon counts')
plt.xlabel('Energy in keV')

plt.subplot(432)
plt.hist(data[2, data[1] == 2], bins=256, range=[0, 255])
plt.title('Channels for PCU2')
plt.ylabel('Photon counts')
plt.xlabel('Energy in keV')

plt.subplot(433)
plt.hist(data[2, data[1] == 3], bins=256, range=[0, 255])
plt.title('Channels for PCU3')
plt.ylabel('Photon counts')
plt.xlabel('Energy in keV')


instrument = np.array(data[1], dtype=int)
instrument[instrument > 0] = 1  # distinguish between PCU0:=0 and PCU1234:=1 energy
channel = np.array(data[2], dtype=int)

# überprüfen, ob das funktioniert
good_data = np.array([data[0], data[1], remapped_energies[instrument+1, channel]])
data = np.array([data[0], data[1], energies[instrument+2, channel]])
# works


energy_bins_0 = np.unique(np.append(energies[2], 130))
energy_bins_23 = np.unique(np.append(energies[3], 120))

plt.subplot(434)
plt.hist(data[2, data[1] == 0], bins=energy_bins_0)
plt.title('Energy for PCU0')
plt.ylabel('Photon counts')
plt.xlabel('Energy in keV')

plt.subplot(435)
plt.hist(data[2, data[1] == 2], bins=energy_bins_23)
plt.title('Energy for PCU2')
plt.ylabel('Photon counts')
plt.xlabel('Energy in keV')

plt.subplot(436)
plt.hist(data[2, data[1] == 3], bins=energy_bins_23)
plt.title('Energy for PCU3')
plt.ylabel('Photon counts')
plt.xlabel('Energy in keV')


plt.subplot(437)
plt.plot(energies[0], energies[2], label='Abs channels PCU0')
plt.plot(energies[0], energies[3], label='Abs channels PCU1234')
plt.legend(loc='upper left')
plt.xlabel('Channel number')
plt.ylabel('Energy in keV')
plt.subplot(438)
plt.plot(np.unique(energies[1]), np.unique(energies[2]), label='Energy levels PCU0')
plt.plot(np.unique(energies[1]), np.unique(energies[3]), label='Energy levels PCU1234')
plt.legend(loc='upper left')
plt.xlabel('Number of Energy level')
plt.ylabel('Energy in keV')
plt.subplot(439)
plt.plot(remapped_energies[0], remapped_energies[1], label='Remapped Energy levels PCU0')
plt.plot(remapped_energies[0], remapped_energies[2], label='Remapped Energy levels PCU1234')
plt.legend(loc='upper left')
plt.xlabel('Channel Number')
plt.ylabel('Energy in keV')


remapped_energy_bins_0 = np.append(np.unique(remapped_energies[1]), 130)
remapped_energy_bins_23 = np.append(np.unique(remapped_energies[2]), 120)

plt.subplot(4, 3, 10)
plt.hist(good_data[2, good_data[1] == 0], bins=remapped_energy_bins_0)
plt.title('Remapped Energy for PCU0')
plt.ylabel('Photon counts')
plt.xlabel('Energy in keV')

plt.subplot(4, 3, 11)
plt.hist(good_data[2, good_data[1] == 2], bins=remapped_energy_bins_23)
plt.title('Remapped Energy for PCU2')
plt.ylabel('Photon counts')
plt.xlabel('Energy in keV')

plt.subplot(4, 3, 12)
plt.hist(good_data[2, good_data[1] == 3], bins=remapped_energy_bins_23)
plt.title('Remapped Energy for PCU3')
plt.ylabel('Photon counts')
plt.xlabel('Energy in keV')

# a)
# knicke in Gerade, weil Energie-Schritte pro Channel ~0.413keV (0.438 für PCU0) sind im Bereich, wo ein Channel einer Energie zugeordnet wird,
# aber wenn zwei Channel einer Energie zugeordnet werden, dann sind die Energie-Schritte doppelt so groß.

# b)
# Vorschlag, korrigiere für dieses Verhalten (was zu den enormen Sprüngen im Histogram führt), indem wir künstliche Energie-Schritte einführen,
# sodass von jedem Energie Channel zum nächsten etwa diese 0.41keV Abstand sind (für die Channel, die auf dieselbe Energie momentan abbilden)
# Anmerkung: mean deltaE von einem Channel zum nächsten steigt an für höhere Energien (PCU23 am anfang 0.41 ist dann 0.44 im Bereich, wo zwei
# channel auf eine Energie abbilden, und ist im dreier Bereich noch größer, Schrittweite ist im Allgemeinen größer für PCU0)

# c)
# Sprünge im Histogram, weil dort zwei nebeneinander liegende Channel laut dem oberen Plot ca. dieselbe Count Anzahl haben,
# aber nun zusammengezählt werden. Somit ist die Anzahl der counts hier doppelt so hoch, wie direkt im Energie bin daneben.
# Das wird auch daran deutlich, dass die Energiebins rechts breiter sind, als bei niedrigen Energien, wegen b)

# d)
# da das remapping wahrscheinlich illegal ist, kann man auch die oberen Energie bins mit einer sensitivity maske überlegen (weights im histogram)


plt.subplots_adjust(left=0.04, right=0.98, hspace=0.5, top=0.96, bottom=0.05)
plt.show()


# plot photon counts histogram over time-energy
start_time = 845
end_time = 1200
time_pix = 2**12
Norm = None  # 'Log'

good_data[0] -= good_data[0].min()
von = np.argmax(good_data[0] > float(start_time))
bis = np.argmax(good_data[0] > float(end_time))
good_data = good_data[:, von:bis]

binned_data = np.zeros(shape=(3, time_pix, remapped_energy_bins_23.size-1))
time_bins = np.zeros(shape=(3, time_pix+1))


binned_data[0], time_bins[0], _ = np.histogram2d(
    good_data[0, good_data[1] == 0], good_data[2, good_data[1] == 0], bins=[time_pix, remapped_energy_bins_0])

binned_data[1], time_bins[1], _ = np.histogram2d(
    good_data[0, good_data[1] == 2], good_data[2, good_data[1] == 2], bins=[time_pix, remapped_energy_bins_23])

binned_data[2], time_bins[2], _ = np.histogram2d(
    good_data[0, good_data[1] == 3], good_data[2, good_data[1] == 3], bins=[time_pix, remapped_energy_bins_23])


plt.subplot(311)
plot_data(binned_data[0], time_bins[0], remapped_energy_bins_0, 'Instrument PCU0', Norm)
plt.ylabel('Energy in keV')
plt.subplot(312)
plot_data(binned_data[1], time_bins[1], remapped_energy_bins_23, 'Instrument PCU2', Norm)
plt.ylabel('Energy in keV')
plt.subplot(313)
plot_data(binned_data[2], time_bins[2], remapped_energy_bins_23, 'Instrument PCU3', Norm)
plt.ylabel('Energy in keV')
plt.subplots_adjust(hspace=0.27, top=0.96, bottom=0.05)
plt.show()


# Different idea:
# Use normal Energy bins, but multiply them by sensitivity mask, to discount Energy bins
# that are filled by two or more channels
# Daniel sagt, ist nicht gewünscht, da die Photon Gesamtanzahl nicht erhalten ist! Eigenes binning einführen!

"""
energy_sensitivity_mask = np.loadtxt("/home/marvin/code/Marvin_Baumann/energy_sensitivity_mask.txt")

data[0] -= data[0].min()
von = np.argmax(data[0] > float(start_time))
bis = np.argmax(data[0] > float(end_time))
data = data[:, von:bis]

binned_data = np.zeros(shape=(3, time_pix, energy_bins_23.size-1))
time_bins = np.zeros(shape=(3, time_pix+1))

print(data.shape)
print(energy_bins_0.shape)
print(energy_bins_23.shape)

binned_data[0], time_bins[0], _ = np.histogram2d(
    data[0, data[1] == 0], data[2, data[1] == 0], bins=[time_pix, energy_bins_0])

binned_data[1], time_bins[1], _ = np.histogram2d(
    data[0, data[1] == 2], data[2, data[1] == 2], bins=[time_pix, energy_bins_23])

binned_data[2], time_bins[2], _ = np.histogram2d(
    data[0, data[1] == 3], data[2, data[1] == 3], bins=[time_pix, energy_bins_23])


# wende sensitivity maske an
binned_data = np.multiply(binned_data, energy_sensitivity_mask)

plt.subplot(321)
plot_data(binned_data[0], time_bins[0], energy_bins_0,
          'Instrument PCU0, Counts weighted with sensitivity mask', Norm)
plt.ylabel('Energy in keV')
plt.subplot(323)
plot_data(binned_data[1], time_bins[1], energy_bins_23,
          'Instrument PCU2, Counts weighted with sensitivity mask', Norm)
plt.ylabel('Energy in keV')
plt.subplot(325)
plot_data(binned_data[2], time_bins[2], energy_bins_23,
          'Instrument PCU3, Counts weighted with sensitivity mask', Norm)
plt.ylabel('Energy in keV')


bar_width_0 = np.array([energy_bins_0[i+1]-E for i, E in enumerate(energy_bins_0[:-1])])
bar_width_23 = np.array([energy_bins_0[i+1]-E for i, E in enumerate(energy_bins_23[:-1])])

plt.subplot(322)
plt.bar(energy_bins_0[:-1], height=np.sum(binned_data[0], axis=0), width=bar_width_0)
plt.title('Energy for PCU0, Counts weighted with sensitivity mask')
plt.ylabel('Photon counts')
plt.xlabel('Energy in keV')

plt.subplot(324)
plt.bar(energy_bins_23[:-1], height=np.sum(binned_data[1], axis=0), width=bar_width_23)
plt.title('Energy for PCU2, Counts weighted with sensitivity mask')
plt.ylabel('Photon counts')
plt.xlabel('Energy in keV')

plt.subplot(326)
plt.bar(energy_bins_23[:-1], height=np.sum(binned_data[2], axis=0), width=bar_width_23)
plt.title('Energy for PCU3, Counts weighted with sensitivity mask')
plt.ylabel('Photon counts')
plt.xlabel('Energy in keV')

plt.subplots_adjust(hspace=0.27, top=0.96, bottom=0.05)
plt.show()
"""


# Ziel: Counts von allen Drei Instrumenten müssen in ein äquidistantes binning überführt werden
# dafür muss ich die vorgegebenen bins für PCU0 und PCU23 jeweils individuell remappen,
# d.h. ein eigenes binning einführen und dann eine Zuordnung von dem binning mit den vorgegebenen
# bins auf mein eigenes binnning bauen.
