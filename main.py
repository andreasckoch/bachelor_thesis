import QPO
#import utilities as QPOutils
#import utilities_old as Oldutils
#import mock_data_s_x as mock_data
import mock_signals
import matplotlib.pyplot as plt
import numpy as np
#import nifty4 as ift

if __name__ == "__main__":

    t_pix = 2**10
    t_volume = 200
    e_pix = 2**10
    e_volume = 127

    # betrachte doppelte Zeit und Energie Dimensionen damit gepaddet werden kann
    s = mock_signals.mock_signal_energy_time(t_pix * 2, t_volume * 2, e_pix * 2, e_volume * 2)

    time_mask = mock_signals.mock_mask(s)
    R = QPO.EnergyTimeResponse(s.domain, time_mask)

    lam = R.times(s)
    s_after = R.adjoint_times(lam)

    x = np.linspace(0, t_volume, num=t_pix)
    start_t = t_volume // 2

    plt.subplot(231)
    plt.imshow(lam.val[0, :, :].T, cmap='inferno', origin='lower', extent=(start_t, t_volume, 0, 256))
    plt.title('PCU0')
    plt.xlabel('Time')
    plt.ylabel('Energy Channels')
    plt.subplot(232)
    plt.imshow(lam.val[1, :, :].T, cmap='inferno', origin='lower', extent=(start_t, t_volume, 0, 256))
    plt.title('PCU2')
    plt.xlabel('Time')
    plt.ylabel('Energy Channels')
    plt.subplot(233)
    plt.imshow(lam.val[2, :, :].T, cmap='inferno', origin='lower', extent=(start_t, t_volume, 0, 256))
    plt.title('PCU3')
    plt.xlabel('Time')
    plt.ylabel('Energy Channels')
    plt.subplot(234)
    plt.imshow(s.val.T, cmap='inferno', origin='lower', extent=(0, 2 * t_volume, 0, 2 * e_volume))
    plt.title('Signal')
    plt.xlabel('Time')
    plt.ylabel('Energy [keV]')
    plt.subplot(235)
    plt.imshow(s_after.val.T, cmap='inferno', origin='lower', extent=(0, 2 * t_volume, 0, 2 * e_volume))
    plt.title('Signal after Responses')
    plt.xlabel('Time')
    plt.ylabel('Energy [keV]')

    #y = np.linspace(0, e_volume, num=e_pix)
    """
    plt.subplot(121)
    plt.bar(x, height=s.val, width=127/e_pix+0.05, bottom=100/t_pix+0.05)
    plt.title('Signal')
    plt.xlabel('Time')
    plt.ylabel('Photon counts')
    plt.subplot(122)
    plt.bar(x, height=lam[:, 0, :].val, width=130/e_pix+0.05, bottom=100/t_pix+0.05)
    plt.title('Signal after Response')
    plt.xlabel('Time')
    plt.ylabel('Photon counts')
    
    plt.subplot(133)
    plt.bar(x, height=s_after.val, width=130/nbins+0.05)
    plt.title('Signal after normal and adjoint Responses')
    plt.xlabel('Time')
    plt.ylabel('Photon counts')
    # plt.xticks(x_ticks)
    # plt.grid(which='both')
    
    
    channels = np.linspace(0, 255, num=256)
    x_ticks = np.linspace(0, 250, num=26)
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
	"""
    plt.subplots_adjust(left=0.04, right=0.99, hspace=0.23, top=0.95, bottom=0.06)
    plt.show()
