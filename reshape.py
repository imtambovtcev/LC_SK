import numpy as np
import sys
import os
import os.path
import time
import matplotlib.pyplot as plt
import magnes
import magnes.graphics
from magnes.utils import state_reshape
import show
import change_anisotropy

load='/media/ivan/64E21891E2186A16/Users/vano2/Documents/LC_SK/initials/cone/cone2/cone_20.npz'
save='/media/ivan/64E21891E2186A16/Users/vano2/Documents/LC_SK/initials/cone/cone2/cone_40.npz'
container = magnes.io.load(load)
system=container.extract_system()
primitives=system.primitives
exchange=system.exchange
representatives=system.representatives
anisotropy=system.anisotropy
bc=system.bc
size=system.size
print(f'{system.size = }')
s = container["STATE"]

new_size=[3,3,40]
s=magnes.utils.state_reshape(s,new_size,[0,0,0])

s0=np.copy(s)
for i in range(0):
    s=np.concatenate([s,s0],axis=0)
#new_size=[33,1,20]
#s=magnes.utils.state_reshape(s,new_size,[-107,0,0])

new_size=s.shape[:3]
print(f'{s.shape = }')
system = magnes.System(primitives, representatives, new_size, bc)
state = system.field3D()
state.upload(s)
s = state.download()
container = magnes.io.container()
container.store_system(system)
container["STATE"] = s
container.save(save)
change_anisotropy.change_anisotropy(save,save,-0.1,5)
show.show(save)



'''
fig,_,_=magnes.graphics.plot_field3D(system,state,slice2D='xy',sliceN=0)
fig.savefig('fj_cut.pdf')
fig,_,_=magnes.graphics.plot_field3D(system,state,slice2D='xz',sliceN=int(new_size[0]/2))
fig.savefig('fj_cut_xz.pdf')
plt.close('all')
'''