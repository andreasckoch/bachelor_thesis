from mock_data_s_xy import mock_signal_s_xy
import numpy as np
import nifty4 as ift
import datetime
import time
import sys
import matplotlib.pyplot as plt
import solver
from plot_data import plot_signal_data as psd

from d4po.problem import Problem

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logpath = 'trash/log_{}.txt'.format(timestamp)
print('Logging to logpath ' + logpath)
logfile = open(logpath, 'w')
plotpath = '/afs/mpa/temp/ankoch/plots'


iterations = 10
t_pix = 2**5 * 2**8  # pixels in time after padding (signal has 2*t_pix pixels)
e_pix = 2**3 * 2**8  # pixels in energy after padding (signal has 2*e_pix pixels)
start_time = 845
end_time = 1245
t_volume = 2**5 * 10  # volume in data
e_volume = 2**3 * 10  # volume in data
smoothing_time = 1.0e-6
smoothing_energy = 1.0e-3
smoothness_sigma_time = 0.3
smoothness_sigma_energy = 0.4


intial_log_message = "Analyzing SGR1806 with:\niterations = {}\nt_pix = 2**{}\ne_pix = {}\nstart_time = {}\nend_time = {}\nt_volume = {}\ne_volume = {}\nsmoothing_time = {:.0e}\nsmoothing_energy = {:.0e}\n"
intial_log_message = intial_log_message.format(iterations, int(np.log2(t_pix)),
                                               e_pix, start_time, end_time, t_volume, e_volume, smoothing_time, smoothing_energy)
print(intial_log_message)
logfile.write(intial_log_message)


def make_problem():

    dic_config = mock_signal_s_xy(t_pix, e_pix, t_volume, e_volume, smoothing_time, smoothing_energy, smoothness_sigma_time, smoothness_sigma_energy, True)

    s = dic_config['signal']
    data = dic_config['data']
    R = dic_config['Response']
    tau_0 = ift.log(dic_config['or_spec_0'])
    tau_1 = ift.log(dic_config['or_spec_1'])
    tau_0_signal = ift.log(dic_config['spec_0'])
    tau_1_signal = ift.log(dic_config['spec_1'])

    psd(s, data, tau_0_signal, tau_1_signal, timestamp, plotpath)

    print("Power Spectra Max:1: {}, 2: {}".format(tau_0.max(), tau_1.max()))

    # initial guesses
    # m_initial = ift.Field(R.domain, val=1.)

    # setting hierarchical parameters for time subdomain
    alpha_0 = ift.Field(tau_0.domain, val=1.)
    q_0 = ift.Field(tau_0.domain, val=1e-12)
    s_0 = smoothing_time

    # setting hierarchical parameters for energy subdomain
    alpha_1 = ift.Field(tau_1.domain, val=1.)
    q_1 = ift.Field(tau_1.domain, val=1e-12)
    s_1 = smoothing_energy

    P = Problem(data, statistics='PLN')
    P.add(s, R=R, Signal_attributes=[[tau_0, alpha_0, q_0, s_0, True],
                                     [tau_1, alpha_1, q_1, s_1, True]])
    # setting starting guess for some fields
    P.tau_uncertainty[0][0].fill(0.)
    P.maps_uncertainty[0].fill(0.)
    P.maps_harmonic_uncertainty[0].fill(0.)
    """
    # draw better starting taus:
    s_dirty = P.ResponseOp[0].adjoint_times(P.data)

    # s smoothen mit FFTSmotthness mit sigma so wählen, dass oszilitationen noch da sind, aber rauschen raus gesmooth ist
    # sigma muss an t_pix angepasst werden.

    S0 = ift.FFTSmoothingOperator(s_dirty.domain, sigma=smoothness_sigma_time, space=0)  # do not confuse smoothing parameters!
    S1 = ift.FFTSmoothingOperator(s_dirty.domain, sigma=smoothness_sigma_energy, space=1)

    s = S1.times(S0.times(s_dirty))

    fft = P.FFTOp[0]
    sk = fft.times(s)
    p0 = ift.power_analyze(sk.integrate(spaces=1))
    p1 = ift.power_analyze(sk.integrate(spaces=0))

    tau_0 = ift.log(p0)
    tau_1 = ift.log(p1)

    # set improved taus
    P.tau = 0, [tau_0, tau_1]
    P.maps = 0, s
    """
    return P, s, tau_0, tau_1


# initilize Problem class
tick = time.time()
P, signal_goal, tau0_goal, tau1_goal = make_problem()
m, s = divmod(time.time()-tick, 60)
h, m = divmod(m, 60)
print('Built Response in %dh%02dmin%02ds.' % (h, m, s))
logfile.write('Built Response in %dh%02dmin%02ds.' % (h, m, s))
sys.stdout.flush()


tack = time.time()
D4PO = solver.D4PO_solver(P, plotpath, timestamp=timestamp)
D4PO(iterations)
m, s = divmod(time.time()-tack, 60)
h, m = divmod(m, 60)
print('Solver took %dh%02dmin%02ds for %d iterations with t_pix=2**%d.' % (h, m, s, iterations, int(np.log2(t_pix))))
logfile.write('Solver took %dh%02dmin%02ds for %d iterations with t_pix=2**%d.' %
              (h, m, s, iterations, int(np.log2(t_pix))))

m, s = divmod(time.time()-tick, 60)
h, m = divmod(m, 60)
print('Total Time: %dh%02dmin%02ds.' % (h, m, s))
logfile.write('Total Time: %dh%02dmin%02ds.' % (h, m, s))
sys.stdout.flush()

P_res = D4PO.results

# mean differences:
s_diff = np.mean(np.absolute(P.maps[0].val - signal_goal.val))
tau0_diff = np.mean(np.absolute(P.tau[0][0].val - tau0_goal.val))
tau1_diff = np.mean(np.absolute(P.tau[0][1].val - tau1_goal.val))
print("Differences: signal: {}, tau0: {}, tau1: {}".format(s_diff, tau0_diff, tau1_diff))
