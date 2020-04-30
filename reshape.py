import numpy as np
import sys
import os
import os.path
import time
import matplotlib.pyplot as plt
import magnes
import magnes.graphics
from magnes.utils import state_reshape

container = magnes.io.load('/home/ivan/LC_SK/initials/matspx10_1_alt.npz' )
system=container.extract_system()
primitives=system.primitives
exchange=system.exchange
representatives=system.representatives
anisotropy=system.anisotropy
bc=system.bc
size=system.size
print(f'{system.size = }')
s = container["STATE"]
s0=np.copy(s)
for i in range(10):
    s=np.concatenate([s,s0],axis=0)
new_size=[301,1,100]
s=magnes.utils.state_reshape(s,new_size,[0,0,0])
print(f'{s.shape = }')
system = magnes.System(primitives, representatives, new_size, bc)
state = system.field3D()
state.upload(s)
s = state.download()
container = magnes.io.container()
container.store_system(system)
container["STATE"] = s
container.save('/home/ivan/LC_SK/initials/matspx10_1_alt_100.npz')
'''
fig,_,_=magnes.graphics.plot_field3D(system,state,slice2D='xy',sliceN=0)
fig.savefig('fj_cut.pdf')
fig,_,_=magnes.graphics.plot_field3D(system,state,slice2D='xz',sliceN=int(new_size[0]/2))
fig.savefig('fj_cut_xz.pdf')
plt.close('all')
'''