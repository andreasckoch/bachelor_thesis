import numpy as np

energy_path = "/home/andi/bachelor/data/arrangeddata/energy_channels.txt"
output_path = "/home/andi/bachelor/data/arrangeddata/energy_channels_remapped.txt"
channels = np.loadtxt(energy_path, usecols=[0, 6, 7], skiprows=25).transpose()

"""
mean = np.zeros(shape=(2, 130))

for i in range(49):
    mean[0, i] = channels[1, 5+i]-channels[1, 4+i]
    mean[1, i] = channels[2, 5+i]-channels[2, 4+i]

for i in range(49, 130):
    mean[0, i] = 0.5*(channels[1, 6+i]-channels[1, 4+i])
    mean[1, i] = 0.5*(channels[2, 6+i]-channels[2, 4+i])
# 54-135

print(mean)
mean = np.mean(mean, axis=1)
print(mean)
# mean ist größer im Bereich, wo zwei Channel einer Energie zugeordnet sind.
"""
for ll in [1, 2]:
    for i, E in enumerate(channels[ll, :]):
        if i < 50 or i > 250:
            continue
        if E == channels[ll, i+5]:
            diff = E - channels[ll, i-1]
            E = channels[ll, i-1]
            diff /= 6.0
            for j in range(5):
                channels[ll, i+j] = (j+1)*diff+E
        elif E == channels[ll, i+4]:
            diff = E - channels[ll, i-1]
            E = channels[ll, i-1]
            diff /= 5
            for j in range(4):
                channels[ll, i+j] = (j+1)*diff+E
        elif E == channels[ll, i+3]:
            diff = E - channels[ll, i-1]
            E = channels[ll, i-1]
            diff /= 4
            for j in range(3):
                channels[ll, i+j] = (j+1)*diff+E
        elif E == channels[ll, i+2]:
            diff = E - channels[ll, i-1]
            E = channels[ll, i-1]
            diff /= 3
            for j in range(2):
                channels[ll, i+j] = (j+1)*diff+E
        elif E == channels[ll, i+1]:
            diff = E - channels[ll, i-1]
            E = channels[ll, i-1]
            diff /= 2
            channels[ll, i] = diff+E
        else:
            continue

print(channels.shape)
# -3 makes left aligned with width=3
np.savetxt(output_path, channels.T, fmt=['%-5d', '%-7.2f', '%-7.2f'], header='Channel PCU0 PCU23')
