# plotte 2D plot (Zeit-Energy) aus SGR1806 Daten mit Photon Counts als Farbverlauf

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import nifty4 as ift
from d4po.problem import Problem
import os
import datetime

import constants as const
import utilities as QPOutils

tau_ticks = 4
start_time = 855
end_time = 1255
t_volume = end_time - start_time  # volume in data
# volume in data is missing in case you need it defined in this file


def main():
    timestamp = '2018-06-05_02-07'
    plotpath = 'results/signal'
    s = np.load(plotpath + '/' + '2018-06-05_02-07-20_40_0_signal.npy')
    #s = f['signal']
    #tau0 = f['tau0']
    #tau1 = f['tau1']
    #tau0k = f['tau0_k']
    #tau1k = f['tau1_k']
    #t_pix = 2**14
    #data = QPOutils.get_data(start_time, end_time, t_pix, seperate_instruments=True)

    plot_results_signal(s, timestamp, plotpath, True)
    #plot_results_data(timestamp, plotpath)


def plot_results_data(timestamp, plotpath, zoom=False):
    start_time = 853
    end_time = 1253
    time_pix = 2**14
    t_length = (end_time-start_time) / time_pix

    data = np.loadtxt(const.data_path, usecols=[0, 2]).transpose()
    data[0] = data[0] - data[0].min()
    von = np.argmax(data[0] > float(start_time))
    bis = np.argmax(data[0] > float(end_time))
    data = data[:, von:bis]
    data, _, _ = np.histogram2d(data[0], data[1], bins=[time_pix, 254])

    if zoom is True:
        start = 0.
        end = 400.
    else:
        start = 0.
        end = t_volume

    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    im = ax.imshow(data.T, cmap='inferno', vmax=7, origin='lower', extent=[0, 400, 0, 255])
    plt.ylabel('Energy channel number')
    plt.xlabel('Time [s]')
    t_ticks = np.linspace(start, end, num=(end-start) // 5)
    ax.set_xticks(t_ticks)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.15)
    plt.colorbar(im, cax=cax)  # extend='max'

    osz_length = 7.5872  # ŕotation frequency: 0.1318
    peak1_length = 1.7
    peak2_length = 2.85
    peak3_length = 1.95
    peak1_data = np.zeros((70, 254), dtype=np.float64)
    peak2_data = np.zeros((117, 254), dtype=np.float64)
    peak3_data = np.zeros((80, 254), dtype=np.float64)
    osz_first = 13.
    for ii in range(47):
        if osz_length * ii + osz_first >= start and osz_length * ii + osz_first + peak1_length + peak2_length <= end:
            x = np.ones((300)) * (osz_length * ii + osz_first)
            y = np.linspace(0, 255, 300)
            ax.plot(x, y, linewidth=0.4, color='red')
            x = np.ones((300)) * (osz_length * ii + osz_first + peak1_length)
            y = np.linspace(0, 255, 300)
            ax.plot(x, y, linewidth=0.3, color='green')
            x = np.ones((300)) * (osz_length * ii + osz_first + peak1_length + peak2_length)
            y = np.linspace(0, 255, 300)
            ax.plot(x, y, linewidth=0.3, color='green')
            x = np.ones((300)) * (osz_length * ii + osz_first + peak1_length + peak2_length + peak3_length)
            y = np.linspace(0, 255, 300)
            ax.plot(x, y, linewidth=0.3, color='green')

            temp1 = data[int((osz_length * ii + osz_first)/t_length):
                         int((osz_length * ii + osz_first + peak1_length)/t_length), :]
            temp2 = data[int((osz_length * ii + osz_first + peak1_length)/t_length):
                         int((osz_length * ii + osz_first + peak1_length + peak2_length)/t_length), :]
            temp3 = data[int((osz_length * ii + osz_first + peak1_length + peak2_length)/t_length):
                         int((osz_length * ii + osz_first + peak1_length + peak2_length + peak3_length)/t_length), :]
            # print(temp1.shape, temp2.shape, temp3.shape)
            peak1_data[:temp1.shape[0], :] += temp1 + np.abs(np.min(temp1))
            peak2_data[:temp2.shape[0], :] += temp2 + np.abs(np.min(temp2))
            peak3_data[:temp3.shape[0], :] += temp3 + np.abs(np.min(temp3))

    save_plot(plotpath, 'data_hist_{}_{}'.format(start, end), timestamp, 0, 0)
    plt.gcf().clear()

    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    x = np.linspace(0, 254, 254)
    ax.plot(x, np.mean(peak1_data, axis=0) / 46, color='darkblue')
    ax.plot(x, np.mean(peak2_data, axis=0) / 46, color='tomato')
    ax.plot(x, np.mean(peak3_data, axis=0) / 46, color='olive')
    ax.set_xlabel('Energy channel number')
    ax.set_ylabel('Photon count')

    save_plot(plotpath, 'data_peaks_summed_{}_{}'.format(start, end), timestamp, 0, 0)
    plt.gcf().clear()


def plot_results_signal(s, timestamp, plotpath, zoom=False):
    t_volume = 400.
    e_volume = 114.6
    e_pix = 256
    shape = s.shape
    t_length = 2 * t_volume / shape[0]
    e_length = 2 * e_volume / shape[1]

    if zoom is True:
        start = 0.
        end = 400.
        e_start = 0.
        e_end = 114.6
    else:
        start = 0.
        end = t_volume
        e_end = e_volume

    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    im = ax.imshow(s[shape[0]//4 + int(start/t_length):shape[0]//4 + int(end/t_length), int(e_start/e_length):int(e_end/e_length)].T,
                   cmap='inferno', origin='lower', vmax=0.001, vmin=-0.0004, extent=[start, end, e_start, e_end])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Energy [keV]')
    # t_ticks = np.linspace(start, end, num=(end-start) // 5)
    # ax.set_xticks(t_ticks)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.15)
    plt.colorbar(im, cax=cax)

    osz_length = 7.5872  # ŕotation frequency: 0.1318
    peak1_length = 1.7
    peak2_length = 2.85
    peak3_length = 1.95
    peak1_signal = np.zeros((140, e_pix), dtype=np.float64)
    peak2_signal = np.zeros((234, e_pix), dtype=np.float64)
    peak3_signal = np.zeros((160, e_pix), dtype=np.float64)
    temp_minima = np.zeros((3,))
    osz_first = 11.
    for ii in range(47):
        if osz_length * ii + osz_first >= start and osz_length * ii + osz_first + peak1_length + peak2_length <= end:
            """
            x = np.ones((200)) * (osz_length * ii + osz_first)
            y = np.linspace(0, e_end, 200)
            ax.plot(x, y, linewidth=1.5, color='red')
            x = np.ones((200)) * (osz_length * ii + osz_first + peak1_length)
            y = np.linspace(0, e_end, 200)
            ax.plot(x, y, linewidth=1.3, color='green')
            x = np.ones((200)) * (osz_length * ii + osz_first + peak1_length + peak2_length)
            y = np.linspace(0, e_end, 200)
            ax.plot(x, y, linewidth=1.3, color='green')
            x = np.ones((200)) * (osz_length * ii + osz_first + peak1_length + peak2_length + peak3_length)
            y = np.linspace(0, e_end, 200)
            ax.plot(x, y, linewidth=1.3, color='green')
            """
            temp1 = s[shape[0]//4 + int((osz_length * ii + osz_first)/t_length):shape[0]//4 + int((osz_length * ii + osz_first + peak1_length)/t_length), int(e_start/e_length):int(e_end/e_length)]
            temp2 = s[shape[0]//4 + int((osz_length * ii + osz_first + peak1_length)/t_length):shape[0]//4 +
                      int((osz_length * ii + osz_first + peak1_length + peak2_length)/t_length), int(e_start/e_length):int(e_end/e_length)]
            temp3 = s[shape[0]//4 + int((osz_length * ii + osz_first + peak1_length + peak2_length)/t_length):shape[0]//4 +
                      int((osz_length * ii + osz_first + peak1_length + peak2_length + peak3_length)/t_length), int(e_start/e_length):int(e_end/e_length)]
            # if temp1.min() < temp_minima[0] and temp2.min() < temp_minima[1] and temp3.min() < temp_minima[2]:
            #    temp_minima = np.abs([temp1.min(), temp2.min(), temp3.min()])

            peak1_signal[:temp1.shape[0], :] += temp1 + np.abs(np.min(temp1))
            peak2_signal[:temp2.shape[0], :] += temp2 + np.abs(np.min(temp2))
            peak3_signal[:temp3.shape[0], :] += temp3 + np.abs(np.min(temp3))

    save_plot(plotpath, 'signal_enhanced_{}_{}'.format(start, end), timestamp, 0, 0)
    plt.gcf().clear()

    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    x = np.linspace(e_start, e_end, e_pix)
    ax.plot(x, np.mean(peak1_signal + temp_minima[0], axis=0) / 46, color='darkblue')
    ax.plot(x, np.mean(peak2_signal + temp_minima[1], axis=0) / 46, color='tomato')
    ax.plot(x, np.mean(peak3_signal + temp_minima[2], axis=0) / 46, color='olive')
    ax.set_xlabel('Energy [keV]')
    ax.set_ylabel('Flux')
    x_ticks = [0, 20, 40, 60, 80, 100]
    ax.set_xticks(x_ticks)

    save_plot(plotpath, 'signal_peaks_summed_{}_{}'.format(start, end), timestamp, 0, 0)
    plt.gcf().clear()


def plot_results_power(tau, domain, k_lengths, timestamp, plotpath, zoom=False, drawLinesMax=False):
    # domain = 0 or 1 [time, or energy]

    plt.figure(figsize=(16, 6))
    ax = plt.gca()

    if zoom is True:
        start = 0.1
        end = 1.5
        mask = k_lengths >= start
        mask *= k_lengths <= end
    else:
        mask = k_lengths > -10

    tau_max = []
    idx = []
    k_list = []
    intervals = [[0.115, 0.15], [0.25, 0.3], [0.37, 0.42], [0.5, 0.6],
                 [0.6, 0.70125], [0.75, 0.85], [0.88, 1.], [1., 1.15125], [1.15125, 1.2], [1.25, 1.35], [1.35, 1.5]]
    for ints in intervals:
        tau_max_prov = np.max(np.exp(tau[k_lengths.tolist().index(ints[0]):k_lengths.tolist().index(ints[1])]))
        idx_prov = np.exp(tau).tolist().index(tau_max_prov)
        tau_max.append(tau_max_prov)
        idx.append(idx_prov)
        k_list.append(k_lengths[idx_prov])

        x = np.ones((200)) * k_lengths[idx_prov]
        y = np.linspace(0.99, 1.01, 200)
        ax.plot(x, y, dashes=[4, 2], color='tomato')

    ax.loglog(k_lengths[mask], np.exp(tau[mask]))
    #y_ticks = [1.0, 1.0001, 1.0002]
    y_ticks = [0.995, 1.0, 1.005, 1.01]
    ax.set_yticks(y_ticks)
    k_list.insert(0, 0.1)
    ax.set_xticks(k_list)
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter(useOffset=False))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter(useOffset=False))

    if domain == 0:
        plt.xlabel('q [1/s]')
        plt.ylabel('P(q)')
    if domain == 1:
        plt.xlabel('k [1/eV]')
        plt.ylabel('P(k)')
    print(np.min(np.exp(tau)), np.max(np.exp(tau)))
    plt.ylim(0.99, 1.01)
    plt.xlim(0.095, 1.5)
    # tick_position = np.exp(np.max(tau0)+np.min(tau0)//2)
    # plt.yticks([1.045, 1.05])
    save_plot(plotpath + '/../final_power_spec', 'tau{}'.format(domain), timestamp, 0, 0)


"""
grad = np.gradient(np.exp(tau[mask]), k_lengths[1] - k_lengths[0])
tau_max = []
if drawLinesMax is True:
    for i, grad_i in enumerate(grad):
        if np.abs(grad_i) < 8e-7 and grad[i-5] > 0 and grad[i+5] < 0:
            tau_max.append([i, k_lengths[mask][i]])
print(tau_max)
"""


def plot_signal():
    t_volume = 400
    e_volume = 114.6
    timestamp = '2018-05-27_23-14-19'
    jj = 5
    ii = 0
    plotpath = 'results/peak_analysis/'

    s = np.load(plotpath + '{}_{}_{}_'.format(timestamp, jj, ii) + 'signal.npy')

    plt.figure(figsize=(8, 4))
    plt.imshow(np.exp(s[s.shape[0]//4:s.shape[0]//4*3, :s.shape[1]//2]).T,
               cmap='inferno', origin='lower', extent=[0, t_volume, 0, e_volume])
    plt.title('Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Energy [keV]')
    plt.colorbar()
    save_plot(plotpath, 'signal', timestamp, jj, ii)
    plt.gcf().clear()


def plot_signal_data(s, data, tau0, tau1, timestamp, plotpath):
    t_volume = s.domain[0].distances[0] * s.domain[0].shape[0]//2
    e_volume = s.domain[1].distances[0] * s.domain[1].shape[0]//2

    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    im = ax.imshow(s.val.T,
                   cmap='inferno', origin='lower', extent=[0, t_volume, 0, e_volume])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Energy [keV]')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.15)
    plt.colorbar(im, cax=cax)
    save_plot(plotpath, 'start_signal', timestamp, 0, 0)
    plt.gcf().clear()

    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    im = ax.imshow(data.val.T,
                   cmap='inferno', origin='lower', extent=[0, t_volume, 0, e_volume])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Energy [keV]')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.15)
    plt.colorbar(im, cax=cax)
    save_plot(plotpath, 'start_data', timestamp, 0, 0)
    plt.gcf().clear()

    plt.figure(figsize=(8, 8))
    plt.loglog(tau0.domain[0].k_lengths, ift.exp(tau0).val)
    plt.title('Time Power Spectrum')
    plt.xlabel('k [1/s]')
    plt.ylabel('P(k)')
    plt.yticks(round_to_1(np.exp(np.linspace(tau0.min(), tau0.max(), num=3))))
    save_plot(plotpath, 'start_tau0', timestamp, 0, 0)
    plt.gcf().clear()

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.loglog(tau1.domain[0].k_lengths, ift.exp(tau1).val)
    y_ticks = [1.5, 2.0]
    ax.set_yticks(y_ticks)
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    plt.title('Energy Power Spectrum')
    ax.set_xlabel('k [1/s]')
    ax.set_ylabel('P(k)')
    # plt.yticks(round_to_1(np.exp(np.linspace(tau1.min(), tau1.max(), num=3))))
    save_plot(plotpath, 'start_tau1', timestamp, 0, 0)
    plt.gcf().clear()

    save_in_files(plotpath, '/{}_{}_{}_'.format(timestamp, 0, 0) + 'start_tau0', tau0.val)
    save_in_files(plotpath, '/{}_{}_{}_'.format(timestamp, 0, 0) + 'start_tau1', tau1.val)


def plot_artefact(save=True):
    # plot artifacts signal from map_2018-04-28_20-05-10.npy
    # with proper axes, times starting from zero, colorbar 3% width, 0.15 pad, etc
    f = np.load('plots/Padding_Kanten/fields_2018-04-28_20-05-10.npz')
    s = f['signal']

    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    shape = s.shape
    im = ax.imshow(s[int(1.05*shape[0]//4):shape[0]//4*3, 0:shape[1]//2].T,
                   cmap='inferno', origin='lower', extent=[0, 400, 0, 127])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Energy [keV]')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.15)
    plt.colorbar(im, cax=cax)
    if save:
        plt.savefig('results/plots/energy_artefacts.png', dpi=1000, bbox_inches='tight')
        plt.gcf().clear()
    else:
        plt.show()


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
    t_volume = P.domain[0][0].distances[0] * P.domain[0][0].shape[0]//2
    e_volume = P.domain[0][1].distances[0] * P.domain[0][1].shape[0]//2
    Pshape = P.maps[0].val.shape

    # signal first
    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    im = ax.imshow(P.maps[0].val[Pshape[0]//4:Pshape[0]//4*3, 0:Pshape[1]//2].T,
                   cmap='inferno', origin='lower', extent=[0, t_volume, 0, e_volume])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Energy [keV]')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.15)
    plt.colorbar(im, cax=cax)
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
    plt.xlabel('q [1/s]')
    plt.ylabel('P(q)')
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

    if jj % 5 == 0:
        save_in_files(plotpath, '/{}_{}_{}_'.format(timestamp, jj, ii) + 'k_lengths_0', P.tau[0][0].domain[0].k_lengths)
        save_in_files(plotpath, '/{}_{}_{}_'.format(timestamp, jj, ii) + 'k_lengths_1', P.tau[0][1].domain[0].k_lengths)

    if jj % 10 == 0:
        save_in_files(plotpath, '/{}_{}_{}_'.format(timestamp, jj, ii) + 'signal', P.maps[0].val)
        save_in_files(plotpath, '/{}_{}_{}_'.format(timestamp, jj, ii) + 'tau0', P.tau[0][0].val)
        save_in_files(plotpath, '/{}_{}_{}_'.format(timestamp, jj, ii) + 'tau1', P.tau[0][1].val)
        # np.save('/afs/mpa/temp/ankoch/results/{}_{}_{}_'.format(timestamp, 5, 0) + 'signal', P.maps[0].val)


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
    # k_lengths_0 = np.load(plotpath + '/k_lengths_0.npy')
    # k_lengths_1 = np.load(plotpath + '/k_lengths_1.npy')

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.loglog(k_lengths_0, np.exp(tau0))
    y_ticks = [1.045, 1.05]
    ax.set_yticks(y_ticks)
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    plt.xlabel('k [1/s]')
    plt.ylabel('P(k)')
    print(np.min(np.exp(tau0)), np.max(np.exp(tau0)))
    plt.ylim(1.042, 1.05)
    # tick_position = np.exp(np.max(tau0)+np.min(tau0)//2)
    # plt.yticks([1.045, 1.05])
    save_plot(plotpath + '/../final_power_spec', 'tau0', timestamp, jj, ii)

    plt.figure(figsize=(8, 8))
    plt.loglog(k_lengths_1, np.exp(tau1))
    plt.title('Reconstructed Power Spectrum in Energy Domain')
    plt.xlabel('k [1/s]')
    plt.ylabel('P(k)')
    # plt.yticks(round_to_1(np.exp(np.linspace(P.tau[0][1].min(), P.tau[0][1].max(), num=tau_ticks))))
    save_plot(plotpath + '/../final_power_spec', 'tau1', timestamp, jj, ii)


if __name__ == "__main__":
    main()
