# plotte 2D plot (Zeit-Energy) aus SGR1806 Daten mit Photon Counts als Farbverlauf

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import nifty4 as ift
from d4po.problem import Problem
import os
import datetime


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

    plt.subplot(221)
    plt.imshow(s.val.T,
               cmap='inferno', origin='lower', extent=[0, t_volume, 0, e_volume])
    plt.title('Signal')
    plt.xlabel('time in s')
    plt.ylabel('Energy in keV')
    plt.colorbar()

    plt.subplot(222)
    plt.imshow(data.val.T,
               cmap='inferno', origin='lower', extent=[0, t_volume, 0, e_volume])
    plt.title('Data')
    plt.xlabel('time in s')
    plt.ylabel('Energy in keV')
    plt.colorbar()

    plt.subplot(223)
    plt.loglog(ift.exp(tau0).val)
    plt.title('Time Power Spectrum')

    plt.subplot(224)
    plt.loglog(ift.exp(tau1).val)
    plt.title('Energy Power Spectrum')

    plt.tight_layout()
    plt.savefig(plotpath + '/start_plot_{}.png'.format(timestamp), dpi=800)
    print('Plotted intermediate plot to ' + plotpath + '/start_plot_{}.png'.format(timestamp))


def powerspec_plot(which_powerspec=0, vmin=None, save=False, etc=None):
    # plot(ift.exp(P.tau))
    pass


def plot_signal(np_array, vmin=None, Norm=None, save=False):

    if Norm == 'Log':
        Norm = colors.LogNorm(vmin=1, vmax=np.max(np_array))
    else:
        Norm = None

    Pshape = np_array.shape
    plt.imshow(np_array[Pshape[0]//4:Pshape[0]//4*3, 0:Pshape[1]//2].T,
               cmap='inferno', vmin=vmin, Norm=Norm, origin='lower', extent=[start_time, end_time, 0, e_volume])
    plt.title('Reconstructed Signal')
    plt.xlabel('time in s')
    plt.ylabel('Energy in keV')
    if save:
        plt.tight_layout()
        plt.savefig('plot.png', dpi=800)
    else:
        plt.subplots_adjust(left=0.04, right=0.98, top=1.0, bottom=0.0)
        plt.show()


def load(filenames, index=0):
    # index=0 means newest result, index=1 second newest, etc
    if len(filenames) > 0:
        file = 'results/'+filenames[index]
        print('Reading:', file)
        return np.load(file)
    else:
        print('No such file.')


def plot_iteration(P, timestamp, jj, plotpath, ii=0, probes=None):
    plt.ioff()
    plt.figure(figsize=(8, 8))
    if probes is None:
        grid = plt.GridSpec(2, 2, wspace=0.2, hspace=0.05, left=0.1, right=0.96, top=0.98, bottom=0.05)
    else:
        grid = plt.GridSpec(3, 2, wspace=0.2, hspace=0.05, left=0.1, right=0.96, top=0.98, bottom=0.05)
    """
    # colorbar
    cmap = plt.cm.get_cmap('viridis')

    for i in np.linspace(0, 1, 11):
        x = [-1, 0, 1]
        y = [i, i, i]
        rgba = cmap(i)
        im = ax.plot(x, y, color=rgba)

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    """
    plt.subplot(grid[0, :])
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
        plt.subplot(grid[1, :])
        plt.imshow(probes.val[Pshape[0]//4:Pshape[0]//4*3, 0:Pshape[1]//2].T,
                   cmap='inferno', origin='lower', extent=[0, t_volume, 0, e_volume])
        plt.title('First Probe taken %d_%d' % (jj, ii))
        plt.xlabel('time in s')
        plt.ylabel('Energy in keV')
        plt.colorbar()
    if probes is None:
        plt.subplot(grid[1, 0])
    else:
        plt.subplot(grid[2, 0])
    plt.loglog(ift.exp(P.tau[0][0]).val)
    plt.title('Reconstructed Time Power Spectrum')
    if probes is None:
        plt.subplot(grid[1, 1])
    else:
        plt.subplot(grid[2, 1])
    plt.loglog(ift.exp(P.tau[0][1]).val)
    plt.title('Reconstructed Energy Power Spectrum')

    plt.tight_layout()
    plt.savefig(plotpath + '/iteration_plot_{}_{}_{}.png'.format(timestamp, jj, ii), dpi=800)
    print('Plotted intermediate plot to ' + plotpath + '/iteration_plot_{}_{}_{}.png'.format(timestamp, jj, ii))


if __name__ == "__main__":
    filenames_fields = get_filenames()

    if len(filenames_fields) > 0:
        fields = load(filenames_fields)
        print('Newest Fields available:', filenames_fields[0].split('fields')[1].split('.')[0][1:], '\n', fields.files)
        print("Plot a field with e.g.: plot(fields['signal'])")
    else:
        print('There are currently no fields files in results/')

"""
    plt.figure(figsize=(8, 8))
    grid = plt.GridSpec(2, 2, wspace=0.15, hspace=0.05, left=0.06, right=0.96, top=0.98, bottom=0.05)
    plt.subplot(grid[0, :])
    Pshape = fields['signal'].shape
    plt.imshow(fields['signal'][Pshape[0]//4:Pshape[0]//4*3, 0:Pshape[1]//2].T,
               cmap='inferno', vmin=-8, origin='lower', extent=[start_time, end_time, 0, e_volume])

    plt.title('Reconstructed Signal')
    plt.xlabel('time in s')
    plt.ylabel('Energy in keV')
    plt.subplot(grid[1, 0])
    plt.loglog(np.exp(fields['tau0']))
    plt.title('Reconstructed Time Power Spectrum')
    plt.subplot(grid[1, 1])
    plt.loglog(np.exp(fields['tau0']))
    plt.title('Reconstructed Energy Power Spectrum')

    # plt.tight_layout()
    plt.savefig('test_plot.png', dpi=800)
    # plt.show()
"""

"""
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
"""
