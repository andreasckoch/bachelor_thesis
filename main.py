import QPO
import utilities as QPOutils
# import utilities_old as Oldutils
# import mock_data_s_x as mock_data
import mock_signals
# import matplotlib.pyplot as plt
# import numpy as np
import nifty4 as ift

if __name__ == "__main__":

    _, energies = QPOutils.get_dicts(return_energies=True, return_channel_fractions=True)
    t_pix = 2**10
    t_volume = 2 * 200
    e_pix = 2 * 129
    e_volume = 2 * 127

    # betrachte doppelte Zeit und Energie Dimensionen damit gepaddet werden kann
    s = mock_signals.mock_signal_energy_time(t_pix, t_volume, e_pix, e_volume)

    R = QPO.EnergyTimeResponse(s.domain)
    time_mask = mock_signals.mock_mask(R._x_new_domain)
    R.set_mask(time_mask)

    ift.extra.consistency_check(R, rtol=1)
    print("All Good!")

    """
    lam = R.times(s)

    s_after = R.adjoint_times(lam.copy())

    x = np.linspace(0, t_volume, num=t_pix)
    start_t = t_volume // 2
    end_t = t_volume // 2 * 3

    plt.subplot(231)
    plt.imshow(lam.val[0, :, :].T, cmap='inferno', origin='lower', extent=(start_t, end_t, 0, 256))
    plt.title('PCU0')
    plt.xlabel('Time')
    plt.ylabel('Energy Channels')
    plt.subplot(232)
    plt.imshow(lam.val[1, :, :].T, cmap='inferno', origin='lower', extent=(start_t, end_t, 0, 256))
    plt.title('PCU2')
    plt.xlabel('Time')
    plt.ylabel('Energy Channels')
    plt.subplot(233)
    plt.imshow(lam.val[2, :, :].T, cmap='inferno', origin='lower', extent=(start_t, end_t, 0, 256))
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
	
    plt.subplots_adjust(left=0.04, right=0.99, hspace=0.23, top=0.95, bottom=0.06)
    plt.show()
	"""
