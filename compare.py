import numpy as np
import matplotlib.pyplot as plt

x = np.load('w.npy')
y = np.load('J001.npy')
y1 = np.load('J011.npy')
plt.plot(x,y,label='[001]')
plt.plot(x,y1,label='[011]')
plt.ylabel('Transmission (vxT)')
plt.xlabel('Frequency (Hz)')
plt.legend()
plt.savefig('comparison.svg')
plt.show()
