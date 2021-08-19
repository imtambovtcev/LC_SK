#!/usr/bin/env python
import numpy as np
import sys
import os
import os.path
import magnes
import magnes.graphics
import magnes.utils
import matplotlib.pyplot as plt
from pathlib import Path
import utilities
import map_file_manager
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
	fig.savefig(directory.joinpath(file + '_xy.pdf'), bbox_inches='tight')
	plt.close('all')
	fig, _, _ = magnes.graphics.plot_field3D(system, state, slice2D='xy', sliceN=0)
	fig.savefig(directory.joinpath(file+'_xy_0.pdf'), bbox_inches='tight')
	plt.close('all')
	if show_extras:
		plt.plot(s[int(s.shape[0]/2),int(s.shape[1]/2),:,0,2])
		plt.xlabel(r'$z$')
		plt.ylabel(r'$m_z$')
		plt.savefig(directory.joinpath(file + '_z.pdf'), bbox_inches='tight')
		plt.close('all')
		plt.plot(s[int(s.shape[0] / 2), int(s.shape[1] / 2), :, 0, 0])
		plt.ylabel(r'$m_x$')
		plt.xlabel(r'$z$')
		plt.savefig(directory.joinpath(file + '_x.pdf'), bbox_inches='tight')
		plt.close('all')

		# toplogical charge
		toplogical_charge=[utilities.toplogical_charge(system,s,i) for i  in range(s.shape[2])]
		plt.plot(toplogical_charge)
		plt.ylabel(r'$Charge$')
		plt.xlabel(r'$z$')
		plt.savefig(directory.joinpath(file + '_tch.pdf'), bbox_inches='tight')
		plt.close('all')

		toplogical_charge_vertical = [utilities.toplogical_charge_vertical(system, s, i) for i in range(s.shape[0])]
		plt.plot(toplogical_charge_vertical)
		plt.ylabel(r'$Charge$')
		plt.xlabel(r'$x$')
		plt.savefig(directory.joinpath(file + '_tch_v.pdf'), bbox_inches='tight')
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

def show(directory,show_extras=True):
	directory=Path(directory)
	if directory.is_dir():
		print(f'{directory = }')
		filelist = [file for file in directory.iterdir() if file.suffix == '.npz']
		print(filelist)
		print(len(filelist))
		for file in filelist:
			plot_npz(file,show_extras=show_extras)
	else:
		file = Path(directory)
		if file.suffix=='.npz':
			plot_npz(file,show_extras=show_extras)

def make_path(data_path,save_file,sort='surf'):
	file_list = [file for file in data_path.iterdir() if file.suffix == '.npz']
	print(file_list)
	if sort == 'surf':
		file_list = sorted(file_list, key=lambda x: -float(x.name.split('_')[-2]))
	else:
		file_list=sorted(file_list,key= lambda x: float(x.name.split('_')[-1][:-4]))

	print(file_list)

	path=[]
	for file in file_list:
		container=magnes.io.load(str(file))
		if 'STATE' in container:
			print()
			ini = container["STATE"]
		else:
			print(f'minimize from 0 image of the path with {container["PATH"].shape = }')
			ini = list(container["PATH"])[0]
		path.append(ini)

	path=np.array(path)

	shapes = np.array([s.shape for s in path])
	xmin = np.min([shapes[:, 0].max(), 100])
	size = [xmin, shapes[:, 1].max(), shapes[:, 2].max()]
	PATH = []
	for s0 in path:
		s = np.copy(s0)
		s = np.concatenate([s for i in range(np.max([size[0] // s.shape[0] + 1, 1]))], axis=0)
		#     if size[0] % s0.shape[0] == 0:
		#         s = np.concatenate([s for i in range(size[0]//s.shape[0])],axis=0)
		#     if size[1] % s0.shape[1] == 0:
		#         s = np.concatenate([s for i in range(size[1]//s.shape[1])],axis=1)
		#     if size[2] % s.shape[2] == 0:
		#         s = np.concatenate([s for i in range(size[2]//s.shape[2])],axis=2)
		s = magnes.utils.state_reshape(s, size, [0, 0, 0])
		PATH.append(s)
	PATH = np.array(PATH)
	print(PATH.shape)
	system = magnes.io.load(str(file_list[0])).extract_system()
	system = magnes.System(system.primitives, system.representatives, size, system.bc)
	container = magnes.io.container('.npz')
	container.store_system(system)
	container['PATH'] = PATH
	container['COMMENT'] = file_list
	container.save(str(save_file))

if __name__ == "__main__":
	directory = './' if len(sys.argv) <= 1 else sys.argv[1:]
	#show_extras = True if len(sys.argv) <= 2 else sys.argv[2]=='True'
	#file=map_file_manager.directory_to_npz(directory)
	[show(d,show_extras='True') for d in directory]
