import numpy as np
import sys

import os

import magnes
import magnes.graphics

import matplotlib.pyplot as plt
import map_file_manager as mfm
import map_info
import map_color

K1=1
K2=10

size = [20,20,20]

Nz=size[2]
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
K = K1 * np.ones(Nz)
K[0] = K1 + K2
K[-1] = K1 + K2
K = K.reshape(1, 1, Nz, 1)
system.add(magnes.Anisotropy(K))
#print(system)
state = system.field3D()

ini=magnes.utils.set_cone(system=system,direction=[0.,0.,1.],period=20,cone = 1, phi0=np.pi/2)
state.upload(ini)
state.satisfy_constrains()
print('z ferr')
print(state.energy_contributions_sum())
s = state.download()
container = magnes.io.container()
container.store_system(system)
container["STATE"] = s
container.save('test/zf.npz')

ini=magnes.utils.set_cone(system=system,direction=[1.,0.,0.],period=20,cone = 1, phi0=np.pi/2)
state.upload(ini)
state.satisfy_constrains()
print('x ferr')
print(state.energy_contributions_sum())
s = state.download()
container = magnes.io.container()
container.store_system(system)
container["STATE"] = s
container.save('test/xf.npz')

ini=magnes.utils.set_cone(system=system,direction=[0.,1.,0.],period=20,cone = 1, phi0=np.pi/2)
state.upload(ini)
state.satisfy_constrains()
print('y ferr')
print(state.energy_contributions_sum())
s = state.download()
container = magnes.io.container()
container.store_system(system)
container["STATE"] = s
container.save('test/yf.npz')

ini=magnes.utils.set_cone(system=system,direction=[0.,0.,1.],period=20,cone = 0, phi0=np.pi/2)
state.upload(ini)
state.satisfy_constrains()
print('z sp')
print(state.energy_contributions_sum())
s = state.download()
container = magnes.io.container()
container.store_system(system)
container["STATE"] = s
container.save('test/zs.npz')

ini=magnes.utils.set_cone(system=system,direction=[1.,0.,0.],period=20,cone = 0, phi0=np.pi/2)
state.upload(ini)
state.satisfy_constrains()
print('x sp')
print(state.energy_contributions_sum())
s = state.download()
container = magnes.io.container()
container.store_system(system)
container["STATE"] = s
container.save('test/xs.npz')

ini=magnes.utils.set_cone(system=system,direction=[0.,1.,0.],period=20,cone = 0, phi0=np.pi/2)
state.upload(ini)
state.satisfy_constrains()
print('y sp')
print(state.energy_contributions_sum())
s = state.download()
container = magnes.io.container()
container.store_system(system)
container["STATE"] = s
container.save('test/ys.npz')



