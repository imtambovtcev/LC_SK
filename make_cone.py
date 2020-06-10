import numpy as np
import sys

import os

import magnes
import magnes.graphics

import matplotlib.pyplot as plt
import map_file_manager as mfm
import map_info
import map_color

#def make_cone(file,save,K1,K2):
#container = magnes.io.load(file)
#ini = container["STATE"]

#size = list(ini.shape[0:3])
size=[1,1,20]
save='/media/ivan/64E21891E2186A16/Users/vano2/Documents/LC_SK/cone/2.npz'
Nz=size[2]
primitives = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
representatives = [(0., 0., 0.)]
bc=[magnes.BC.PERIODIC,magnes.BC.PERIODIC,magnes.BC.FREE]

K_bulk=0.0
K_surf=10
J=1.
D=np.tan(np.pi/10)


system = magnes.System(primitives, representatives, size, bc)
origin = magnes.Vertex(cell=[0, 0, 0])
system.add(magnes.Exchange(origin, magnes.Vertex([1, 0, 0]), J, [D, 0., 0.]))
system.add(magnes.Exchange(origin, magnes.Vertex([0, 1, 0]), J, [0., D, 0.]))
system.add(magnes.Exchange(origin, magnes.Vertex([0, 0, 1]), J, [0., 0., D]))
system.add(magnes.Anisotropy(np.power(D,2)*K_bulk))
K = np.zeros(Nz)
K[0] = K_surf
K[-1] = K_surf
K =np.power(D,2)*K.reshape(1, 1, Nz, 1)
system.add(magnes.Anisotropy(K,axis=[1,0,0]))
print(system)
state = system.field3D()
ini=magnes.utils.set_cone(system=system,direction=[0.,0.,1.],period=size[2]/2,cone = 0, phi0=0)
state.upload(ini)
state.satisfy_constrains()
print(state.energy_contributions_sum())
s = state.download()
container = magnes.io.container()
container.store_system(system)
container["STATE"] = s
container.save(save)
'''
if __name__ == "__main__":
    make_cone(sys.argv[1],sys.argv[2],float(sys.argv[3]),float(sys.argv[4]))
'''