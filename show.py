#!/usr/bin/env python
import numpy as np
import sys
import os
os.environ['MAGNES_BACKEND'] = 'numpy'
import os.path
import magnes
import magnes.graphics
import magnes.utils
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import CubicSpline


def plot_state(system,s,directory,file,show_extras=False):
	directory=Path(directory)
	state = system.field3D()
	state.upload(s)
	fig, _, _ = magnes.graphics.plot_field3D(system, state, slice2D='xz', sliceN=int(system.size[1] / 2))
	fig.savefig(directory.joinpath(file+'_xz.pdf'), bbox_inches='tight')
	plt.close('all')
	fig, _, _ = magnes.graphics.plot_field3D(system, state, slice2D='yz', sliceN=int(system.size[0] / 2))
	fig.savefig(directory.joinpath(file+'_yz.pdf'), bbox_inches='tight')
	plt.close('all')
	fig, _, _ = magnes.graphics.plot_field3D(system, state, slice2D='xy', sliceN=int(system.size[2] / 2))
	fig.savefig(directory.joinpath(file+'_xy.pdf'), bbox_inches='tight')
	plt.close('all')
	if show_extras:
		plt.plot(s[int(s.shape[0]/2),int(s.shape[1]/2),:,0,2])
		plt.xlabel(r'$z$')
		plt.ylabel(r'$m_z$')
		plt.savefig(directory.joinpath(file + '_z.pdf'), bbox_inches='tight')
		plt.close('all')
		plt.plot(s[int(s.shape[0] / 2), int(s.shape[1] / 2), :, 0, 1])
		plt.ylabel(r'$m_x$')
		plt.xlabel(r'$z$')
		plt.savefig(directory.joinpath(file + '_x.pdf'), bbox_inches='tight')
		plt.close('all')

def plot_npz(file,show_extras=False):
	try:
		file=Path(file)
		print(file)
		container = magnes.io.load(str(file))
		system = container.extract_system()
		if 'STATE' in container:
			s = np.array(container["STATE"])
			plot_state(system=system, s=s, directory=file.parent, file=file.stem,show_extras=show_extras)
		else:
			print(container['PATH'].shape)
			print(container['PATH'].shape[0])
			if container['PATH'].shape[0] == 1:
				s = list(container["PATH"])[0]
				plot_state(system=system, s=s, directory=file.parent, file=file.stem,show_extras=show_extras)
			else:
				for idx, s in enumerate(container['PATH']):
					plot_state(system=system, s=s, directory=file.parent.joinpath(file.stem), file=file.stem + '_{}'.format(idx),show_extras=show_extras)

	except:
		print('failed')

def show(directory):
	directory=Path(directory)
	if directory.is_dir():
		print(f'{directory = }')
		filelist = [file for file in directory.iterdir() if file.suffix == '.npz']
		print(filelist)
		print(len(filelist))
		for file in filelist:
			plot_npz(file)
	else:
		file = Path(directory)
		if file.suffix=='.npz':
			plot_npz(file,show_extras=True)


if __name__ == "__main__":
    directory = './' if len(sys.argv) <= 1 else sys.argv[1]
    show(directory)
