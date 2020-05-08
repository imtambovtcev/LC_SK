import numpy as np
import sys

import os
os.environ['MAGNES_BACKEND'] = 'numpy'
import magnes
import magnes.graphics
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
import map_file_manager as mfm
import map_info
import map_color

#def turn(file,save,K1,K2):

'''def basis_from_vector(v,dv):
    v1=v/np.linalg.norm(v)
    v2=dv/np.linalg.norm(dv)
    v3 = np.cross(v1, v2)
    v3 /= np.linalg.norm(v3)
    v2=np.cross(v3,v1)
    v2/=np.linalg.norm(v2)
    return np.array([v1,v2,v3])

def rotation_matrix(v1,v2):
    dv=v2-v1
    r = R.from_rotvec(np.pi/2 * np.array([0, 0, 1]))
'''

def rotate(base2,result1,base1):
    angle=np.arccos(np.dot(result1,base1))
    #print(f'{angle = }')
    if np.abs(angle)<0.01:
        return base2
    elif np.abs(angle-np.pi)<0.01:
        return -base2
    else:
        cross = np.cross(result1, base1)
        return np.dot(R.from_rotvec(angle*cross/np.linalg.norm(cross)).as_matrix(),base2)


base1='/home/ivan/LC_SK/initials/cone.npz'
base2='/home/ivan/LC_SK/initials/skyrmion.npz'
result1='/home/ivan/LC_SK/initials/tilted_cut.npz'
result2='/home/ivan/LC_SK/test.npz'

save='spx/skyrmion.npz'
K1=-0.1
K2=7.5

container = magnes.io.load(base1)
base1_state = container["STATE"]
container = magnes.io.load(base2)
base2_state = container["STATE"]
container = magnes.io.load(result1)
result1_state = container["STATE"]
assert base1_state.shape==base2_state.shape
assert base1_state.shape==result1_state.shape

size= list(base1_state.shape)[:3]

result2_state=np.full(result1_state.shape,np.nan)

for ct0 in range(size[0]):
    for ct1 in range(size[1]):
        for ct2 in range(size[2]):
            result2_state[ct0,ct1,ct2,0,:]=rotate(base2_state[ct0,ct1,ct2,0,:],base1_state[ct0,ct1,ct2,0,:],result1_state[ct0,ct1,ct2,0,:])

print(size)

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
state.upload(result2_state)
state.satisfy_constrains()
print(state.energy_contributions_sum())
s = state.download()
container = magnes.io.container()
container.store_system(system)
container["STATE"] = s
container.save(result2)
