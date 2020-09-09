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


def print_anisotropy(file):
	file = Path(file)
	print(file)
	container = magnes.io.load(str(file))
	system = container.extract_system()
	print(system)
	for ani in system.anisotropy:
		try:
			print(f'{ani.strength} = ')
			D=system.exchange[0].DM[0,0,0,0]
			print(f'{D = }')
			print(f'{ani.strength/(D*D)} = ')
			Kb=ani.strength.reshape(-1)[1]/(D*D)
			Ks=ani.strength.reshape(-1)[0]/(D*D)-Kb
			print('K_b/(D^2)={}\tK_s/(D^2)={}'.format(Kb,Ks))
		except:
			print('error')

if __name__ == "__main__":
	print_anisotropy(sys.argv[1])
