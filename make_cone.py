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
size=[199,1,1]
save='/home/ivan/LC_SK/initials/matspx10_1_1.npz'
Nz=size[2]
primitives = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
representatives = [(0., 0., 0.)]
bc=[magnes.BC.PERIODIC,magnes.BC.PERIODIC,magnes.BC.PERIODIC]
K1=0
K2=0
J=1.
D=np.tan(np.pi/10)
K1*=(D**2)
K2 *= (D ** 2)

system = magnes.System(primitives, representatives, size, bc)
origin = magnes.Vertex(cell=[0, 0, 0])
system.add(magnes.Exchange(origin, magnes.Vertex([1, 0, 0]), J, [D, 0., 0.]))
system.add(magnes.Exchange(origin, magnes.Vertex([0, 1, 0]), J, [0., D, 0.]))
system.add(magnes.Exchange(origin, magnes.Vertex([0, 0, 1]), J, [0., 0., D]))
K = K1 * np.ones(Nz)
K[0] = K1 + K2
K[-1] = K1 + K2
K = K.reshape(1, 1, Nz, 1)
system.add(magnes.Anisotropy(K))
print(system)
state = system.field3D()
ini=magnes.utils.set_cone(system=system,direction=[1.,0.,0.],period=size[0]/10,cone = 0.0, phi0=np.pi/2)
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