import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

DIRECTORY = '2018-04-26_20-51-25'
start_time = 845
end_time = 1245
t_volume = end_time - start_time  # volume in data
e_volume = 127  # volume in data

signal = np.load('../signal/2018-04-26_20-51-25/signal.npy')
Pshape = signal.shape
print(np.min(signal), np.max(signal))
Norm = colors.LogNorm(vmin=1, vmax=np.max(signal))
plt.imshow(signal[Pshape[0]//4:Pshape[0]//4*3, :Pshape[1]//2].T,
           cmap='inferno', vmin=1e-3, norm=Norm, origin='lower', extent=[start_time, end_time, 0, e_volume])
plt.title('Reconstructed Signal')
plt.xlabel('time in s')
plt.ylabel('Energy in keV')

"""
tau0 = np.load('../signal/'+DIRECTORY+'/tau0.npy')
tau1 = np.load('../signal/'+DIRECTORY+'/tau1.npy')
tau0_uncer = np.load('../signal/'+DIRECTORY+'/tau0_uncer.npy')
tau1_uncer = np.load('../signal/'+DIRECTORY+'/tau1_uncer.npy')

plt.subplot(221)
plt.loglog(np.exp(tau0).T)
plt.xlabel('v [1/s]')
plt.ylabel('P(v)')
plt.subplot(222)
plt.loglog(np.exp(tau1).T)
plt.xlabel('v [1/s]')
plt.ylabel('P(v)')
plt.subplot(223)
plt.plot(np.exp(tau0_uncer).T)
plt.xlabel('v [1/s]')
plt.ylabel('P(v)')
plt.subplot(224)
plt.plot(np.exp(tau1_uncer).T)
plt.xlabel('v [1/s]')
plt.ylabel('P(v)')


t_pix = 2**18  # pixels in time after padding (signal has 2*t_pix pixels)
e_pix = 256  # pixels in energy after padding (signal has 2*e_pix pixels)
start_time = 845
end_time = 1245
t_volume = end_time - start_time  # volume in data
e_volume = 127  # volume in data


# time space
x_0 = ift.RGSpace(2*t_pix, distances=t_volume/t_pix)
k_0 = x_0.get_default_codomain()
p_0 = ift.PowerSpace(harmonic_partner=k_0)

# energy space
x_1 = ift.RGSpace(2*e_pix, distances=e_volume/e_pix)
k_1 = x_1.get_default_codomain()
p_1 = ift.PowerSpace(harmonic_partner=k_1)

ift.plot(ift.log(ift.Field(ift.RGSpace(tau0.shape, distances=k_0.distances), val=tau0)), name='./trash/tau0.png')
"""


plt.plot()
# plt.savefig('../signal/'+DIRECTORY+'/PowerSpectra.png')
plt.savefig('../signal/'+DIRECTORY+'/signal.png')
