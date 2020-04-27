import os
import sys

import magnes
import magnes.graphics
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from scipy.ndimage.filters import gaussian_filter1d

print(sys.argv[1])

directory="./" if len(sys.argv)<=1 else sys.argv[1]

filelist = [file for file in os.listdir(directory) if file[-4:]== '.npz' ]
l=[]
for i in range(len(filelist)):
#	if i%120<12:
        l.append(filelist[i])
filelist=l
print(len(filelist))

data=[]

for file_from_list in filelist:
    result_directory=file_from_list[:-4]+'/'
    if not os.path.exists(directory+result_directory):
        os.makedirs(directory+result_directory)
    print(directory+file_from_list)
    container = magnes.io.load(directory+file_from_list)
    system = container.extract_system()
    path_np = np.array(container["PATH"])
    energy=[]
    state = system.field3D()
    for ct,i in enumerate(path_np):
        state.upload(i)
        fig, _, _ = magnes.graphics.plot_field3D(system, state, slice2D='xz', sliceN=int(i.shape[0] / 2))
        fig.savefig(directory+result_directory+str(ct)+'_xz.pdf', bbox_inches='tight')
        plt.close('all')
        energy.append(state.energy_contributions_sum()['total'])
        container2 = magnes.io.container()
        container2.store_system(system)
        container2["STATE"] = i
        container2.save(directory+result_directory +str(ct)+'.npz')
    energy=np.array(energy)
    energy=energy-energy.min()
    print(energy)
    np.savez(directory + result_directory + 'energy.npz',energy=energy)
    plt.plot(energy)
    plt.xlabel('state number', fontsize=16)
    plt.ylabel('$E$', fontsize=16)
    plt.savefig(directory + result_directory + 'energy.pdf')
    plt.show()
    ysmoothed = gaussian_filter1d(energy, sigma=0.9)
    plt.plot(np.array(range(len(energy))), ysmoothed)
    plt.xlabel('state number', fontsize=16)
    plt.ylabel('$E$', fontsize=16)
    plt.savefig(directory + result_directory + 'energy_int.pdf')
    plt.show()

    print(f'max at {argrelextrema(energy, np.greater)},\twith {energy[argrelextrema(energy, np.greater)]}')
    print(f'min at {argrelextrema(energy, np.less)},\twith {energy[argrelextrema(energy, np.less)]}')