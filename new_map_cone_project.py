import numpy as np

import os

os.environ['MAGNES_BACKEND'] = 'nocache'
os.environ['MAGNES_DTYPE'] = 'f32'

import magnes
import magnes.graphics

import matplotlib.pyplot as plt
import map_file_manager as mfm
import map_info
import map_color

file='/media/ivan/64E21891E2186A16/LC_SK/cone/cone_3.npz'
directory='/media/ivan/64E21891E2186A16/LC_SK/cone/paper/cone_3_map/'
state_name='cone'


if not os.path.exists(directory):
    os.makedirs(directory)

container = magnes.io.load(file)
ini = list(container["PATH"])[0]



J=1.0;

size=ini.shape[0:3]
print(size)
SZ=ini.shape[1]
Nz=ini.shape[2]
primitives = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
representatives = [(0., 0., 0.)]
bc=[magnes.BC.PERIODIC,magnes.BC.PERIODIC,magnes.BC.FREE]



Klist,Kaxis=mfm.file_manager(directory,
					   params={'double':False, 'add': [np.round(np.linspace(0., 3., 61), decimals=6).tolist(),
													   np.round(np.linspace(20, 60, 11), decimals=6).tolist()]})

#'source':'/media/ivan/64E21891E2186A16/Users/vano2/Documents/LC_SK/spx/xsp_map/best/'})

Kx0=1000
Kz0=-1000

for idx,Kv in enumerate(Klist,start=1):
	system = magnes.System(primitives, representatives, size, bc)
	origin = magnes.Vertex(cell=[0, 0, 0])
	D = np.tan(np.pi / Kv[1])
	system.add(magnes.Exchange(origin, magnes.Vertex([1, 0, 0]), J, [D, 0., 0.]))
	system.add(magnes.Exchange(origin, magnes.Vertex([0, 1, 0]), J, [0., D, 0.]))
	system.add(magnes.Exchange(origin, magnes.Vertex([0, 0, 1]), J, [0., 0., D]))
	system.add(magnes.Anisotropy(np.power(D, 2) * Kv[0]))
	Kx = np.zeros(Nz)
	Kx[0] = Kx0
	Kx[-1] = Kx0
	Kx = np.power(D, 2)*Kx.reshape(1, 1, Nz, 1)
	system.add(magnes.Anisotropy(Kx, axis=[1, 0, 0]))
	Kz = np.zeros(Nz)
	Kz[0] = Kz0
	Kz[-1] = Kz0
	Kz = np.power(D, 2) * Kz.reshape(1, 1, Nz, 1)
	system.add(magnes.Anisotropy(Kz, axis=[0, 0, 1]))
	print(system)
	state=system.field3D()
	state.upload(ini)
	state.satisfy_constrains()

	plot=False
	maxtime=6000
	alpha=0.1
	precision=5e-5
	catcher=magnes.EveryNthCatcher(1000)
	reporters=[magnes.TextStateReporter()]
	if plot and system.number_of_spins<1000000:
		reporters.append(magnes.graphics.GraphStateReporter())
		reporters.append(magnes.graphics.VectorStateReporter3D(slice2D='xy',sliceN=int(Nz/2)))
		reporters.append(magnes.graphics.VectorStateReporter3D(slice2D='xy',sliceN=0))
		reporters.append(magnes.graphics.VectorStateReporter3D(slice2D='xz',sliceN=int(SZ/2)))
	minimizer=magnes.StateNCG(system, reference=None, stepsize=alpha, maxiter=None, maxtime=200, precision=precision, reporter=magnes.MultiReporter(reporters), catcher=catcher)

	minimizer.optimize(state)
	s=state.download()
	container = magnes.io.container(directory +state_name+ '_{:.5f}_{:.5f}.npz'.format(Kv[0], Kv[1]))
	container.store_system(system)
	container["PATH"] = [s]
	container.save(directory +state_name+ '_{:.5f}_{:.5f}.npz'.format(Kv[0], Kv[1]))

map_info.map_info(directory)
map_color.map_color(directory)
