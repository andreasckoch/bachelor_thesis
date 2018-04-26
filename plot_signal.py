import numpy as np
import matplotlib.pyplot as plt

signal = np.load('../signal/signal_2018-04-26_00-35-50.npy')

plt.imshow(np.exp(signal).T, cmap='inferno', norm='normalize')
plt.plot()
