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
from pathlib import Path

import utilities

load=Path('/home/ivan/LC_SK/lcsim/D=20/1400_from_nothing_3.npz')
save=Path('/home/ivan/LC_SK/lcsim/D=20/1400_view.npz')
n=2 #multiplyer

container = magnes.io.load(str(load))
system=container.extract_system()
primitives=system.primitives
exchange=system.exchange
representatives=system.representatives
anisotropy=system.anisotropy
bc=system.bc
size=system.size
print(f'{system.size = }')
path=container["PATH"]
print(f'{path.shape = }')
path= list(path)
new_size=(np.array(size)-1)*n+1
print(f'{new_size = }')
path=utilities.smooth_path_reshape(path,new_size)
new_size=path.shape[1:4]
print(f'{new_size = }')
system = magnes.System(primitives, representatives, new_size, bc)
save_container = magnes.io.container(str(save))
save_container.store_system(system)
save_container['PATH'] = np.array(path)
for item in container.keys():
    if item not in ['PATH','Path','STATE','State','SYSTEM','System']:
        print(f'item {item} added')
        save_container[item]=container[item]
save_container.save(str(save))
#change_anisotropy.change_anisotropy(save,save,-0.1*np.power(np.tan(np.pi / 10),2),5*np.power(np.tan(np.pi / 10),2))
#show.show(str(save))
