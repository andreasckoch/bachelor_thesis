import numpy as np
import nifty4 as ift
import utilities as QPOutils
import QPO
import Solver
import matplotlib.pyplot as plt

from d4po.problem import Problem

start_time = 845
end_time = 1245
t_pix = 2 * 2**10
t_volume = 2 * (end_time - start_time)
e_pix = 2 * 127
e_volume = 2 * 127

# time space
x_0 = ift.RGSpace(t_pix, distances=t_volume / t_pix)
k_0 = x_0.get_default_codomain()
p_0 = ift.PowerSpace(harmonic_partner=k_0)

# energy space
x_1 = ift.RGSpace(e_pix, distances=e_volume / e_pix)
k_1 = x_1.get_default_codomain()
p_1 = ift.PowerSpace(harmonic_partner=k_1)

# Power Spectra ###
# Time ##
# setting signal parameters
lambda_s_0 = 2.5  # signal correlation length
sigma_s_0 = 4.5  # signal variance

# calculating parameters
total_volume_0 = t_volume
k0_0 = 4. / (2 * np.pi * lambda_s_0)
a_s_0 = sigma_s_0 ** 2. * lambda_s_0 * total_volume_0

# creating Power Operator with given spectrum
spec_s_0 = (lambda k: a_s_0 / (1 + (k / k0_0) ** 2) ** 2)
tau_0 = ift.log(ift.Field(p_0, val=spec_s_0(p_0.k_lengths), dtype=np.float64))

# Energy ##
# setting signal parameters
lambda_s_1 = .2  # signal correlation length
sigma_s_1 = .3  # signal variance

# calculating parameters
total_volume_1 = e_volume
k0_1 = 4. / (2 * np.pi * lambda_s_1)
a_s_1 = sigma_s_1 ** 2. * lambda_s_1 * total_volume_1

# creating Power Operator with given spectrum
spec_s_1 = (lambda k: a_s_1 / (1 + (k / k0_1) ** 2) ** 2)
tau_1 = ift.log(ift.Field(p_1, val=spec_s_1(p_1.k_lengths), dtype=np.float64))

# Build Response ###
signal_domain = ift.DomainTuple.make((x_0, x_1))
R = QPO.Response(signal_domain)

# Load Data ###
data = QPOutils.get_data(start_time, end_time, t_pix//2, seperate_instruments=True)
time_mask = QPOutils.get_time_mask(data, R.time_padded_domain, threshold=2)
data = ift.Field(R.target, val=data)  # np.clip(data, 1e-10, data.max()))

# time mask
R.set_mask(time_mask)

# initial guesses
m_initial = ift.Field(R.domain, val=0.5)

# setting hierarchical parameters into 0-th subdomain
alpha_0 = ift.Field(p_0, val=1.)
q_0 = ift.Field(p_0, val=1e-12)
s_0 = 1.

# setting hierarchical parameters into 1-st subdomain
alpha_1 = ift.Field(p_1, val=1.)
q_1 = ift.Field(p_1, val=1e-12)
s_1 = 1.

# initilize Problem class
P = Problem(data, statistics='PLN')
P.add(m_initial, R=R, Signal_attributes=[[tau_0, alpha_0, q_0, s_0, True],
                                         [tau_1, alpha_1, q_1, s_1, True]])
print(P.maps[0].min())

start_t = t_volume // 4
end_t = t_volume // 4 * 3

data = P.data

D4PO = Solver.D4PO_solver(P)
D4PO(3)

P_res = D4PO.results

print(P_res.maps[0].min())
plt.imshow(P_res.maps[0].val[t_pix//4:t_pix//4*3, :e_pix//2].T,
           cmap='inferno', vmin=0., origin='lower', extent=(start_t, end_t, 0, e_volume // 2))
plt.title('Signal Reconstruction')
plt.xlabel('Time')
plt.ylabel('Energy [keV]')

plt.subplots_adjust(left=0.04, right=0.99, hspace=0.23, top=0.95, bottom=0.06)
plt.show()
