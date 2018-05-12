import numpy as np
import nifty4 as ift
import datetime
import time
import sys
import matplotlib.pyplot as plt

import utilities as QPOutils
import QPO
import solver
from plot_data import plot_signal_data as psd

from d4po.problem import Problem

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logpath = 'trash/log_{}.txt'.format(timestamp)
print('Logging to logpath ' + logpath)
logfile = open(logpath, 'w')
plotpath = 'trash'


iterations = 5
t_pix = 2**8  # pixels in time after padding (signal has 2*t_pix pixels)
e_pix = 256  # pixels in energy after padding (signal has 2*e_pix pixels)
start_time = 845
end_time = 1245
t_volume = end_time - start_time  # volume in data
e_volume = 114.6  # volume in data
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

    # time space
    x_0 = ift.RGSpace(2*t_pix, distances=t_volume/t_pix)
    k_0 = x_0.get_default_codomain()
    p_0 = ift.PowerSpace(harmonic_partner=k_0)

    # energy space
    x_1 = ift.RGSpace(2*e_pix, distances=e_volume/e_pix)
    k_1 = x_1.get_default_codomain()
    p_1 = ift.PowerSpace(harmonic_partner=k_1)

    ## Time ##
    # setting signal parameters
    lambda_s_0 = 2.5  # signal correlation length
    sigma_s_0 = 4.5  # signal variance

    # calculating parameters
    total_volume_0 = t_volume
    k0_0 = 4. / (2 * np.pi * lambda_s_0)
    a_s_0 = sigma_s_0 ** 2. * lambda_s_0 * total_volume_0

    # creating Power Operator with given spectrum
    spec_s_0 = (lambda k: a_s_0 / (1 + (k / k0_0) ** 2))
    tau_0 = ift.log(ift.Field(p_0, val=spec_s_0(p_0.k_lengths), dtype=np.float64))

    ## Energy ##
    # setting signal parameters
    lambda_s_1 = .2  # signal correlation length
    sigma_s_1 = .3  # signal variance

    # calculating parameters
    total_volume_1 = e_volume
    k0_1 = 4. / (2 * np.pi * lambda_s_1)
    a_s_1 = sigma_s_1 ** 2. * lambda_s_1 * total_volume_1

    # creating Power Operator with given spectrum
    spec_s_1 = (lambda k: a_s_1 / (1 + (k / k0_1) ** 2))
    tau_1 = ift.log(ift.Field(p_1, val=spec_s_1(p_1.k_lengths), dtype=np.float64))

    ### Build Response ###############################
    signal_domain = (x_0, x_1)
    R = QPO.Response(signal_domain)

    ### Load Data ####################################
    data = QPOutils.get_data(start_time, end_time, t_pix, seperate_instruments=True)
    time_mask = QPOutils.get_time_mask(data, R.time_padded_domain)
    R.set_mask(time_mask)
    data = ift.Field(R.target, val=np.clip(data, 1e-10, data.max()))

    # initial guesses
    m_initial = ift.Field(R.domain, val=1.)

    # setting hierarchical parameters for time subdomain
    alpha_0 = ift.Field(tau_0.domain, val=1.2)
    q_0 = ift.Field(tau_0.domain, val=1e-12)
    s_0 = smoothing_time

    # setting hierarchical parameters for energy subdomain
    alpha_1 = ift.Field(tau_1.domain, val=1.2)
    q_1 = ift.Field(tau_1.domain, val=1e-12)
    s_1 = smoothing_energy

    P = Problem(data, statistics='PLN')
    P.add(m_initial, R=R, Signal_attributes=[[tau_0, alpha_0, q_0, s_0, True],
                                             [tau_1, alpha_1, q_1, s_1, True]])

    psd(m_initial, data, tau_0, tau_1, timestamp, plotpath)
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
    return P


# initilize Problem class
tick = time.time()
P = make_problem()
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


# results speichern:
try:
    P_res.dump('results/r_{}.p'.format(timestamp))
except Exception as e:
    print(e, 'not able to save to trash, instead saving locally')
    P_res.dump('r_{}.p'.format(timestamp))
# nachher wieder laden mit Problem.load(), dann plotten
