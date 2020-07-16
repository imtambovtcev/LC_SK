import os
import sys
import magnes
import magnes.graphics
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from scipy.ndimage.filters import gaussian_filter1d
from pathlib import Path


def path(file,show=False):
    assert file.suffix == '.npz'
    save_dir = file.parent.joinpath(file.stem)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    container = magnes.io.load(str(file))
    system = container.extract_system()
    path=[system.field3D() for state in container["PATH"]]
    path =magnes.path.Path([p.upload(state) for p,state in zip(path,container["PATH"])])
    energy_from_ref=path.energy(reference=path[-1].energy())
    energy = path.energy()
    size=container["PATH"][0].shape
    nspin=size[0]*size[1]*size[2]
    epu=energy/nspin
    epu_from_ref= energy_from_ref / nspin
    dist=path.distances()

    np.savez(str(save_dir.joinpath('energy.npz')), dist=dist, dist_norm=dist/dist.max(), energy=energy, energy_from_ref=energy_from_ref, epu=epu,
             epu_from_ref=epu_from_ref)

    plt.plot(energy_from_ref,'.')
    plt.xlabel('State number', fontsize=16)
    plt.ylabel('Energy', fontsize=16)
    plt.savefig(str(save_dir.joinpath('energy.pdf')))
    if show: plt.show()
    plt.close('all')

    ysmoothed = gaussian_filter1d(energy_from_ref, sigma=0.9)
    plt.plot(np.array(range(len(energy_from_ref))), ysmoothed)
    plt.xlabel('State number', fontsize=16)
    plt.ylabel('$E$', fontsize=16)
    plt.savefig(str(save_dir.joinpath('energy_int.pdf')))
    if show: plt.show()
    plt.close('all')

    plt.plot(dist,energy_from_ref,'.')
    plt.xlabel('Distance', fontsize=16)
    plt.ylabel('Energy', fontsize=16)
    plt.savefig(str(save_dir.joinpath('energy_dist.pdf')))
    if show: plt.show()
    plt.close('all')

    print(f'max at {argrelextrema(energy_from_ref, np.greater)},\twith {energy_from_ref[argrelextrema(energy_from_ref, np.greater)]}')
    print(f'min at {argrelextrema(energy_from_ref, np.less)},\twith {energy_from_ref[argrelextrema(energy_from_ref, np.less)]}')

def show_path(directory,show=False):
    print(sys.argv[1])
    if os.path.isdir(directory):
        filelist = [file for file in os.listdir(directory) if file[-4:]== '.npz' ]
        for file_from_list in filelist:
            path(Path(file_from_list),show=show)
    elif directory.suffix == '.npz':
        path(directory,show=show)
    else:
        print('Invalid input')


if __name__ == "__main__":
    directory = './' if len(sys.argv) <= 1 else sys.argv[1]
    show = True if len(sys.argv) <= 2 else sys.argv[2]=='True'
    show_path(Path(directory),show=show)