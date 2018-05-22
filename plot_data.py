# plotte 2D plot (Zeit-Energy) aus SGR1806 Daten mit Photon Counts als Farbverlauf

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import nifty4 as ift
from d4po.problem import Problem
import os
import datetime

tau_ticks = 4
start_time = 845
end_time = 1245
t_volume = end_time - start_time  # volume in data
# volume in data is missing in case you need it defined in this file


def get_filenames(file='fields'):
    filenames = os.listdir('results/')
    filenames = [f for f in filenames if f.split('_')[0] == file]
    dates = [f.split(file)[1].split('.')[0][1:] for f in filenames if f.split('_')[0] == file]
    dates = [datetime.datetime.strptime(f, "%Y-%m-%d_%H-%M-%S") for f in dates]

    # this is me, implementing the simple SelectionSort Algorithm
    n = len(filenames)
    for i in range(n):
        large = i
        for j in range(i+1, n):
            if(dates[j] > dates[large]):
                large = j
        tempf = filenames[large]
        tempd = dates[large]
        filenames[large] = filenames[i]
        dates[large] = dates[i]
        filenames[i] = tempf
        dates[i] = tempd

    return filenames


def plot_signal_data(s, data, tau0, tau1, timestamp, plotpath):
    t_volume = s.domain[0].distances[0] * s.domain[0].shape[0]//2
    e_volume = s.domain[1].distances[0] * s.domain[1].shape[0]//2

    plt.figure(figsize=(8, 4))
    plt.imshow(s.val.T,
               cmap='inferno', origin='lower', extent=[0, t_volume, 0, e_volume])
    plt.title('Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Energy [keV]')
    plt.colorbar()
    save_plot(plotpath, 'start_signal', timestamp, 0, 0)
    plt.gcf().clear()

    plt.figure(figsize=(8, 4))
    plt.imshow(data.val.T,
               cmap='inferno', origin='lower', extent=[0, t_volume, 0, e_volume])
    plt.title('Data')
    plt.xlabel('Time [s]')
    plt.ylabel('Energy [keV]')
    plt.colorbar()
    save_plot(plotpath, 'start_data', timestamp, 0, 0)
    plt.gcf().clear()

    plt.figure(figsize=(8, 8))
    plt.loglog(ift.exp(tau0).val)
    plt.title('Time Power Spectrum')
    plt.xlabel('k [1/s]')
    plt.ylabel('P(k)')
    plt.yticks(round_to_1(np.exp(np.linspace(tau0.min(), tau0.max(), num=tau_ticks))))
    save_plot(plotpath, 'start_tau0', timestamp, 0, 0)
    plt.gcf().clear()

    plt.figure(figsize=(8, 8))
    plt.loglog(tau1.domain[0].k_lengths, ift.exp(tau1).val)
    plt.title('Energy Power Spectrum')
    plt.xlabel('k [1/s]')
    plt.ylabel('P(k)')
    plt.yticks(round_to_1(np.exp(np.linspace(tau1.min(), tau1.max(), num=tau_ticks))))
    save_plot(plotpath, 'start_tau1', timestamp, 0, 0)
    plt.gcf().clear()


def round_to_1(x):
    prov = []
    for x_i in x:
        if x_i != 0:
            if x_i < 1:
                decimal = -np.log10(np.absolute(x_i)).astype('i8') + 1
                prov.append(np.round(x_i, decimal))
            else:
                decimal = -np.log10(np.absolute(x_i)).astype('i8')
                prov.append(np.round(x_i, decimal))
    return prov


def plot_iteration(P, timestamp, jj, plotpath, ii=0, probes=None):
    plt.ioff()
    plt.figure(figsize=(8, 8))
    # grid = plt.GridSpec(2, 2, wspace=0.2, hspace=0.05, left=0.1, right=0.96, top=0.98, bottom=0.05)

    plt.subplot(221)
    Pshape = P.maps[0].val.shape
    t_volume = P.domain[0][0].distances[0] * P.domain[0][0].shape[0]//2
    e_volume = P.domain[0][1].distances[0] * P.domain[0][1].shape[0]//2
    plt.imshow(P.maps[0].val[Pshape[0]//4:Pshape[0]//4*3, 0:Pshape[1]//2].T,
               cmap='inferno', origin='lower', extent=[0, t_volume, 0, e_volume])
    plt.title('Reconstructed Signal for iteration step %d_%d' % (jj, ii))
    plt.xlabel('time in s')
    plt.ylabel('Energy in keV')
    plt.colorbar()
    if probes is not None:
        plt.subplot(222)
        plt.imshow(probes.val[Pshape[0]//4:Pshape[0]//4*3, 0:Pshape[1]//2].T,
                   cmap='inferno', origin='lower', extent=[0, t_volume, 0, e_volume])
        plt.title('First Probe taken %d_%d' % (jj, ii))
        plt.xlabel('time in s')
        plt.ylabel('Energy in keV')
        plt.colorbar()

    plt.subplot(223)
    plt.loglog(ift.exp(P.tau[0][0]).val)
    plt.title('Reconstructed Time Power Spectrum')
    # plt.yticks

    plt.subplot(224)
    plt.loglog(ift.exp(P.tau[0][1]).val)
    plt.title('Reconstructed Energy Power Spectrum')

    plt.tight_layout()

    try:
        plt.savefig(plotpath + '/iteration_plot_{}_{}_{}.png'.format(timestamp, jj, ii), dpi=800)
        print('Plotted intermediate plot to ' + plotpath + '/iteration_plot_{}_{}_{}.png'.format(timestamp, jj, ii))
    except Exception as e:
        plt.savefig('fallback/iteration_plot_{}_{}_{}.png'.format(timestamp, jj, ii), dpi=800)
        print('Plotted intermediate plot to fallback/iteration_plot_{}_{}_{}.png'.format(timestamp, jj, ii))


def real_plot_iteration(P, timestamp, jj, plotpath, ii=0, probes=None):

    # signal first
    plt.ioff()
    plt.figure(figsize=(8, 4))
    Pshape = P.maps[0].val.shape
    t_volume = P.domain[0][0].distances[0] * P.domain[0][0].shape[0]//2
    e_volume = P.domain[0][1].distances[0] * P.domain[0][1].shape[0]//2
    plt.imshow(P.maps[0].val[Pshape[0]//4:Pshape[0]//4*3, 0:Pshape[1]//2].T,
               cmap='inferno', origin='lower', extent=[0, t_volume, 0, e_volume])
    plt.title('Reconstructed Signal for iteration step %d_%d' % (jj, ii))
    plt.xlabel('Time [s]')
    plt.ylabel('Energy [keV]')
    plt.colorbar()
    save_plot(plotpath, 'signal', timestamp, jj, ii)
    plt.gcf().clear()

    # probe 2nd
    if probes is not None:
        plt.figure(figsize=(8, 4))
        plt.imshow(probes.val[Pshape[0]//4:Pshape[0]//4*3, 0:Pshape[1]//2].T,
                   cmap='inferno', origin='lower', extent=[0, t_volume, 0, e_volume])
        plt.title('First Probe taken %d_%d' % (jj, ii))
        plt.xlabel('Time [s]')
        plt.ylabel('Energy [keV]')
        plt.colorbar()
        save_plot(plotpath, 'probe', timestamp, jj, ii)
        plt.gcf().clear()

    # Power Spectra
    plt.figure(figsize=(8, 8))
    plt.loglog(P.tau[0][0].domain[0].k_lengths, ift.exp(P.tau[0][0]).val)
    plt.title('Reconstructed Power Spectrum in Time Domain')
    plt.xlabel('k [1/s]')
    plt.ylabel('P(k)')
    save_plot(plotpath, 'tau0', timestamp, jj, ii)
    plt.gcf().clear()

    plt.figure(figsize=(8, 8))
    plt.loglog(P.tau[0][1].domain[0].k_lengths, ift.exp(P.tau[0][1]).val)
    plt.title('Reconstructed Power Spectrum in Energy Domain')
    plt.xlabel('k [1/s]')
    plt.ylabel('P(k)')
    save_plot(plotpath, 'tau1', timestamp, jj, ii)
    plt.gcf().clear()

    # save power spectra in files
    save_in_files(plotpath, '/{}_{}_{}_'.format(timestamp, jj, ii) + 'tau0', P.tau[0][0].val)
    save_in_files(plotpath, '/{}_{}_{}_'.format(timestamp, jj, ii) + 'tau1', P.tau[0][1].val)

    if jj % 5 == 0:
        save_in_files(plotpath, '/{}_{}_{}_'.format(timestamp, jj, ii) + 'k_lengths_0', P.tau[0][0].domain[0].k_lengths)
        save_in_files(plotpath, '/{}_{}_{}_'.format(timestamp, jj, ii) + 'k_lengths_1', P.tau[0][1].domain[0].k_lengths)


def save_in_files(plotpath, name, array):
    # save some array in a file
    try:
        np.save(plotpath + name, array)
    except Exception as e:
        np.save('fallback/' + name, array)


def save_plot(plotpath, name, timestamp, jj, ii):
    try:
        plt.savefig(plotpath + '/{}_{}_{}_'.format(timestamp, jj, ii) + name + '.png', dpi=800, bbox_inches='tight')
        print('Plotted intermediate plot to ' + plotpath + '/{}_{}_{}_'.format(timestamp, jj, ii) + name + '.png')
    except Exception as e:
        plt.savefig('fallback/' + '{}_{}_{}_'.format(timestamp, jj, ii) + name + '.png', dpi=800)
        print('Plotted intermediate plot to fallback/' + '{}_{}_{}_'.format(timestamp, jj, ii) + name + '.png')


def plot_power_from_file(timestamp, jj, ii, plotpath):
    tau0 = np.load(plotpath + '/{}_{}_{}_tau0'.format(timestamp, jj, ii) + '.npy')
    tau1 = np.load(plotpath + '/{}_{}_{}_tau1'.format(timestamp, jj, ii) + '.npy')
    k_lengths_0 = np.load(plotpath + '/{}_{}_{}_k_lengths_0'.format(timestamp, 0, ii) + '.npy')
    k_lengths_1 = np.load(plotpath + '/{}_{}_{}_k_lengths_1'.format(timestamp, 0, ii) + '.npy')
    #k_lengths_0 = np.load(plotpath + '/k_lengths_0.npy')
    #k_lengths_1 = np.load(plotpath + '/k_lengths_1.npy')

    plt.figure(figsize=(8, 8))
    plt.loglog(k_lengths_0, np.exp(tau0))
    plt.title('Reconstructed Power Spectrum in Time Domain')
    plt.xlabel('k [1/s]')
    plt.ylabel('P(k)')
    print(np.min(np.exp(tau0)), np.max(np.exp(tau0)))
    tick_position = np.exp(np.max(tau0)+np.min(tau0)//2)
    plt.yticks([tick_position])
    save_plot(plotpath + '/../final_power_spec', 'tau0', timestamp, jj, ii)

    plt.figure(figsize=(8, 8))
    plt.loglog(k_lengths_1, np.exp(tau1))
    plt.title('Reconstructed Power Spectrum in Energy Domain')
    plt.xlabel('k [1/s]')
    plt.ylabel('P(k)')
    # plt.yticks(round_to_1(np.exp(np.linspace(P.tau[0][1].min(), P.tau[0][1].max(), num=tau_ticks))))
    save_plot(plotpath + '/../final_power_spec', 'tau1', timestamp, jj, ii)


if __name__ == "__main__":
    plot_power_from_file("2018-05-21_23-54-11", 9, 0, 'results/5_mock')
