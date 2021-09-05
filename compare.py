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
plt.figure()

J2d = np.load('J2d.npy')
plt.imshow(J2d.T,origin='lower',extent=(0,10,0,90),aspect='auto',interpolation='bicubic')
plt.xlabel('Frequency (THz)')
plt.ylabel('Angle (degree)')
plt.savefig('J2d.svg')
plt.show()
