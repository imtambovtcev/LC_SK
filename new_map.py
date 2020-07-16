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
from minimize import *
from pathlib import Path

'''
file='/home/ivan/LC_SK/initials/cone2Ku0Ks5.npz'
directory='/home/ivan/LC_SK/spx/mat2_cone/'
state_name='cone'
'''
file=Path('/media/ivan/64E21891E2186A16/Users/vano2/Documents/LC_SK/initials/cone/cone1/20.npz')
directory=Path('/media/ivan/64E21891E2186A16/Users/vano2/Documents/LC_SK/new_spx/alt_20_bulk/cone/')
state_name='cone'

'''
file='/home/ivan/LC_SK/initials/cone.npz'
directory='/home/ivan/LC_SK/skt/cone/'
state_name='cone'

file='/home/ivan/LC_SK/initials/toron.npz'
directory='/home/ivan/LC_SK/skt/toron/'
state_name='toron'

file='/home/ivan/LC_SK/initials/skyrmion.npz'
directory='/home/ivan/LC_SK/skt/skyrmion/'
state_name='skyrmion'
'''

if not os.path.exists(directory):
    os.makedirs(directory)

container = magnes.io.load(str(file))
ini = container["STATE"]

J=1.0;D=np.tan(np.pi/10)

size=ini.shape[0:3]
print(size)
SZ=ini.shape[1]
Nz=ini.shape[2]
primitives = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
representatives = [(0., 0., 0.)]
bc=[magnes.BC.PERIODIC,magnes.BC.PERIODIC,magnes.BC.FREE]

Klist,Kaxis=mfm.file_manager(directory,
					   params={'source':'/media/ivan/64E21891E2186A16/Users/vano2/Documents/LC_SK/new_spx/alt_20_bulk/1/best'})
'''						   'double':False,
							   'add': [np.round(np.linspace(0, 10, 1), decimals=6).tolist(),
                                             np.round(np.linspace(0, 40, 11), decimals=6).tolist()]})
				'''

for idx,Kv in enumerate(Klist,start=1):
	system,s,energy=minimize(ini=ini,J=J,D=D,Kbulk=np.power(D, 2)*Kv[0],Ksurf=np.power(D, 2)*Kv[1])
	container = magnes.io.container()
	container.store_system(system)
	container["STATE"] = s
	container['ENERGY']=energy
	container.save(str(directory.joinpath(state_name+ '_{:.5f}_{:.5f}.npz'.format(Kv[0], Kv[1]))))

map_info.map_info(directory)
#map_color.map_color(directory)
