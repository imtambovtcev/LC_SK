import numpy as np
import sys

import os

import magnes
import magnes.graphics

import matplotlib.pyplot as plt
import map_file_manager as mfm
import map_info
import map_color

#def turn(file,save,K1,K2):

file='initials/ferr_sk_alt.npz'
save='spx/skyrmion.npz'
K1=1.5
K2=-10

container = magnes.io.load(file)
ini = container["STATE"]
print(ini.shape)
ini = np.array(np.moveaxis(ini,2,0))
ini=np.array(np.moveaxis(np.array([ini[:,:,:,:,2],ini[:,:,:,:,1],ini[:,:,:,:,0]]),0,-1))
ini=magnes.utils.state_reshape(state=ini,new_shape=[20,40,40],x0=[0,-30,-30])
size = list(ini.shape[0:3])

print(ini.shape)
print(size)
Nz=ini.shape[2]
primitives = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
representatives = [(0., 0., 0.)]
bc=[magnes.BC.PERIODIC,magnes.BC.PERIODIC,magnes.BC.FREE]

J=1.
D=np.tan(np.pi/10)
K1*=(D**2)
K2 *= (D ** 2)

system = magnes.System(primitives, representatives, size, bc)
origin = magnes.Vertex(cell=[0, 0, 0])
system.add(magnes.Exchange(origin, magnes.Vertex([1, 0, 0]), J, [D, 0., 0.]))
system.add(magnes.Exchange(origin, magnes.Vertex([0, 1, 0]), J, [0., D, 0.]))
system.add(magnes.Exchange(origin, magnes.Vertex([0, 0, 1]), J, [0., 0., D]))
K = np.zeros(size[2])
K[0] = K2
K[-1] = K2
K = K.reshape(1, 1, size[2],1)
system.add(magnes.Anisotropy(K))
system.add(magnes.Anisotropy(K1,axis=[1.,0.,0.]))
print(system)
state = system.field3D()
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
    turn(sys.argv[1],sys.argv[2],float(sys.argv[3]),float(sys.argv[4]))
'''