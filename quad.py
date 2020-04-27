import numpy as np
import sys
import os
import os.path
import time
import matplotlib.pyplot as plt
import magnes
import magnes.graphics
import magnes.utils

container = magnes.io.load('initials/quad.npz')
system=container.extract_system()
primitives=system.primitives
exchange=system.exchange
representatives=system.representatives
anisotropy=system.anisotropy
bc=system.bc
size=system.size
print(system.size)
s = container["STATE"]
s=magnes.utils.state_reshape(s,[50,50,20],[-25,-25,0])
s1=magnes.utils.state_reshape(s,[50,50,20],[15,15,0])
s2=magnes.utils.state_reshape(s,[50,50,20],[-15,15,0])
sb=np.concatenate((s1,s2),axis=0)
s1=magnes.utils.state_reshape(s,[50,50,20],[15,-15,0])
s2=magnes.utils.state_reshape(s,[50,50,20],[-15,-15,0])
st=np.concatenate((s1,s2),axis=0)
s=np.concatenate((sb,st),axis=1)
s=magnes.utils.state_reshape(s,[140,140,20],[20,20,0])
print(s.shape)
system = magnes.System(primitives, representatives, s.shape[:3], bc)
state = system.field3D()
for i in exchange:
    system.add(i)
for i in anisotropy:
    system.add(magnes.Anisotropy(i.strength))
state.upload(s)
s = state.download()
container = magnes.io.container()
container.store_system(system)
container["STATE"] = s
container.save('new_skyrmion.npz')
fig,_,_=magnes.graphics.plot_field3D(system,state,slice2D='xy',sliceN=0)
fig.savefig('new_state.pdf')
fig,_,_=magnes.graphics.plot_field3D(system,state,slice2D='xz',sliceN=int(s.shape[1]/2))
fig.savefig('new_state_xz.pdf')
plt.close('all')
