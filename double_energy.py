import numpy as np
import sys
import os
import os.path
import time
import matplotlib.pyplot as plt
import magnes
import magnes.graphics
import magnes.utils

def coordinate(s,ground):
    diff=s-ground
    diff[diff[:,:,:,:,2]<=0]=[0.,0.,0.]
    diff = diff[:,:,:,:,2]
    diff=diff/np.sum(diff)
    r0=np.moveaxis(np.moveaxis(np.array(np.meshgrid(np.linspace(0,1,s.shape[0]),np.linspace(0, 1, s.shape[1]),np.linspace(0, 1, s.shape[2]))), 0, -1),1,0)
    r=r0*diff
#    plt.imshow(r[:,:,10,2])
#    plt.show()
    return (np.sum(r,axis=(0,1,2))-0.5)*s.shape[0:3]

def double_energy(directory):
    os.environ['MAGNES_BACKEND'] = 'numpy'
    if not os.path.exists(directory+'info/'):
        os.makedirs(directory+'info/')
    filelist = [file for file in os.listdir(directory) if len(file)>3 and file[-4:]== '.npz' and len(str.split(file, '_'))==3]
    state_name=np.array([str.split(file,'_')[0] for file in filelist])
    container = magnes.io.load(directory + 'ini.npz')
    system = container.extract_system()
    ground = container["STATE"]
    state = system.field3D()
    state.upload(ground)
    ground_energy = state.energy_contributions_sum()['total']
    assert np.all(state_name == state_name[0])
    xyz = np.full([len(filelist), 3], np.nan)
    energy = np.full(len(filelist), np.nan)
    for idx,file in enumerate(filelist):
        container = magnes.io.load(directory + file)
        s = container["STATE"]
        state.upload(s)
        xyz[idx]=coordinate(s,ground)
        energy[idx]=state.energy_contributions_sum()['total']

    np.savez(directory+'info/Energy.npz',coordinate=xyz,energy=energy,ground_energy=ground_energy)

if __name__ == "__main__":
    directory = './' if len(sys.argv) <= 1 else sys.argv[1]
    double_energy(directory)