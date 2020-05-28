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


def show(directory):
	if os.path.isdir(directory):
		filelist = [file for file in os.listdir(directory) if len(file)>3 and file[-4:] == '.npz']
		print(len(filelist))
		for file in filelist:
			try:
				print(directory+file)
				container = magnes.io.load(directory+file)
				system = container.extract_system()
				s = np.array(container["STATE"])
				state = system.field3D()
				state.upload(s)
				fig, _, _ = magnes.graphics.plot_field3D(system, state, slice2D='xz', sliceN=int(system.size[1] / 2))
				fig.savefig(directory+file[:-4]+'xz.pdf', bbox_inches='tight')
				plt.close('all')
				fig, _, _ = magnes.graphics.plot_field3D(system, state, slice2D='yz', sliceN=int(system.size[0] / 2))
				fig.savefig(directory+file[:-4]+'yz.pdf', bbox_inches='tight')
				plt.close('all')
				fig, _, _ = magnes.graphics.plot_field3D(system, state, slice2D='xy', sliceN=int(system.size[2] / 2))
				fig.savefig(directory+file[:-4]+'xy.pdf', bbox_inches='tight')
				plt.close('all')
			except:
				print('failed')
	else:
		file = Path(directory)
		if file.suffix=='.npz':
			try:
				container = magnes.io.load(directory)
				system = container.extract_system()
				s = np.array(container["STATE"])
				state = system.field3D()
				state.upload(s)
				fig, _, _ = magnes.graphics.plot_field3D(system, state, slice2D='xz', sliceN=int(system.size[1] / 2))
				fig.savefig(directory[:-4] + 'xz.pdf', bbox_inches='tight')
				plt.close('all')
				fig, _, _ = magnes.graphics.plot_field3D(system, state, slice2D='yz', sliceN=int(system.size[0] / 2))
				fig.savefig(directory[:-4] + 'yz.pdf', bbox_inches='tight')
				plt.close('all')
				fig, _, _ = magnes.graphics.plot_field3D(system, state, slice2D='xy', sliceN=int(system.size[2] / 2))
				fig.savefig(directory[:-4] + 'xy.pdf', bbox_inches='tight')
				plt.close('all')
				plt.plot(s[:,int(s.shape[1]/2),int(s.shape[2]/2),0,2])
				plt.show()
			except:
				print('failed')


if __name__ == "__main__":
    directory = './' if len(sys.argv) <= 1 else sys.argv[1]
    show(directory)
