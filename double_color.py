import numpy as np
import sys
import os
import os.path
import time
import matplotlib.pyplot as plt
from scipy import interpolate

directory='skyrmion_distance/'

file=np.load(directory+'info/Energy.npz')
xy=file['coordinate'][:,:2]
energy=file['energy']-file['ground_energy']
x=np.array(sorted(list(set(xy[:, 0]))))
y=np.array(sorted(list(set(xy[:, 1]))))

marker_size=15
plt.scatter(xy[:,0], xy[:,1], marker_size, c=energy)
plt.savefig(directory+'info/Energyscatter.pdf')
plt.title("Energy")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.show()

linfit=interpolate.interp2d(xy[:,0],xy[:,1],energy)
x_plot=np.linspace(x.min(),x.max(),100)
y_plot=np.linspace(y.min(),y.max(),100)
xy_plot=np.moveaxis(np.meshgrid(x_plot,y_plot),0,-1)
square_energy=linfit(x_plot,y_plot).reshape(xy_plot.shape[0],xy_plot.shape[1])
plt.contourf(xy_plot[:,:,0],xy_plot[:,:,1],square_energy)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig(directory+'info/Energy.pdf')
plt.show()
r=np.linalg.norm(xy,axis=1)
plt.plot(r,energy,'.')
plt.xlabel('r')
plt.ylabel('Energy')
plt.tight_layout()
plt.savefig(directory+'info/Energy_R.pdf')
plt.show()