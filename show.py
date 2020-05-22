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
				centre=s[:,int(system.size[1] / 2),int(system.size[2] / 2),0,1]
				qm = s[:, int(system.size[1] / 2), int(system.size[2]/2)-3, 0, 1]
				qp = s[:, int(system.size[1] / 2), int(system.size[2]/2) + 3, 0, 1]
				sp_c=CubicSpline(range(len(centre)),centre)
				sp_qm=CubicSpline(range(len(qm)),qm)
				sp_qp = CubicSpline(range(len(qp)), qp)
				roots_c=sp_c.derivative().roots().tolist()
				roots_qm = sp_qm.derivative().roots().tolist()
				roots_qp = sp_qp.derivative().roots().tolist()
				roots_c=np.array([i for i in roots_c if i>=0 and sp_c(i)>0.8*centre.max() and sp_c(i)<1.2*centre.max()])
				roots_qm = np.array(
					[i for i in roots_qm if i>=0 and sp_qm(i) > 0.8 * qm.max() and sp_qm(i) < 1.2 * qm.max()])
				roots_qp = np.array(
					[i for i in roots_qp if i>=0 and sp_qp(i) > 0.8 * qp.max() and sp_qp(i) < 1.2 * qp.max()])
				print(roots_c[2]-roots_qm[2])
				print(roots_c[2] - roots_qp[2])
				print(roots_qm[2] - roots_qp[2])
				print(f'{np.arctan(6/(roots_qm[2] - roots_qp[2]))*360/(2*np.pi) = }')
				plt.plot(roots_c,sp_c(roots_c),'bx')
				plt.plot(roots_qm, sp_qm(roots_qm), 'gx')
				plt.plot(roots_qp, sp_qp(roots_qp),'rx')
				plt.plot(np.linspace(0,len(centre),400),sp_c(np.linspace(0,len(centre),400)),'b')
				plt.plot(np.linspace(0, len(qm), 400), sp_qm(np.linspace(0, len(qm), 400)),'g')
				plt.plot(np.linspace(0, len(qm), 400), sp_qm(np.linspace(0, len(qm), 400)),'r')
				plt.plot(centre,'b.')
				plt.plot(qm, 'g.')
				plt.plot(qp,'r.')
				plt.show()
			except:
				print('failed')


if __name__ == "__main__":
    directory = './' if len(sys.argv) <= 1 else sys.argv[1]
    show(directory)
