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


def path_minimize(file, save):
    save = Path(save)
    container = magnes.io.load(str(file))
    system = container.extract_system()
    path=[system.field3D() for s in container["PATH"]]
    path = [p.upload(s) for p, s in zip(path, container["PATH"])]
    path=magnes.Path(path)
    path.mep()
    container = magnes.io.container('.npz')
    container.store_system(system)
    container["PATH"] = path.download()
    container.save(str(save))

def make_tilted_path(file1,file2,save,m=1.,N=20):
    save=Path(save)
    container1 = magnes.io.load(str(file1))
    system = container1.extract_system()
    s1= container1["PATH"][0]
    container2 = magnes.io.load(str(file2))
    s2 = container2["PATH"][0]
    state=system.field3D()
    s3=s1+np.linspace(1,0,s1.shape[0]).reshape(-1,1)*np.array([0,0,m]).reshape(1,-1)
    state.upload(s3)
    state.satisfy_constrains()
    s3=state.download()
    path=[system.field3D(), system.field3D(), system.field3D()]
    path=[p.upload(s) for p,s in zip(path,[s1,s3,s2])]
    path = magnes.Path(path)
    path=path.interpolate(relative_positions=np.linspace(0, 1,N))
    container = magnes.io.container('.npz')
    container.store_system(system)
    container["PATH"] = path.download()
    container.save(str(save))


def make_path(file1,file2,save,N=20):
    save = Path(save)
    container1 = magnes.io.load(str(file1))
    system = container1.extract_system()
    s1 = container1["PATH"][0]
    container2 = magnes.io.load(str(file2))
    s2 = container2["PATH"][0]
    path = [system.field3D(), system.field3D()]
    path = [p.upload(s) for p, s in zip(path, [s1, s2])]
    path = magnes.Path(path)
    path = path.interpolate(relative_positions=np.linspace(0, 1, N))
    container = magnes.io.container('.npz')
    container.store_system(system)
    container["PATH"] = path.download()
    container.save(str(save))

if __name__ == "__main__":
    make_path(sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]))