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
from ai import cs


def path_minimize(file, save):
    save = Path(save)
    container = magnes.io.load(str(file))
    system = container.extract_system()
    path=[system.field3D() for s in container["PATH"]]
    path = [p.upload(s) for p, s in zip(path, container["PATH"])]
    maxtime = 6000
    alpha = 0.1
    precision = 5e-5
    catcher = magnes.EveryNthCatcher(1000)
    reporters = [magnes.TextStateReporter()]
    minimizer = magnes.StateGDwM(system, reference=None, stepsize=alpha, maxiter=None, maxtime=200, precision=precision,
                                 reporter=magnes.MultiReporter(reporters), catcher=catcher)
    path[0]=minimizer.optimize(path[0])
    path[-1] = minimizer.optimize(path[-1])
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

def make_cone_path(file1,file2,save,N=20):
    save=Path(save)
    container1 = magnes.io.load(str(file1))
    system = container1.extract_system()
    s0= container1["PATH"][0]
    size=s0.shape
    path=[]
    path_len=N-1
    for n in range(N):
        state=system.field3D()
        theta=np.pi*(1+n/path_len)*(1-np.linspace(0,1,size[2]))-np.pi
        phi_0=4*n*(path_len-n)/(path_len*path_len) #(0,1) 1 at n=path_len/2
        phi=np.arcsin(phi_0*(1-np.linspace(0,1,size[2])))
        print(f'{theta.min() = },{theta.max() = },{np.sin(phi.min()) = },{np.sin(phi.max()) = }')
        x, y, z = cs.sp2cart(r=1, theta=phi, phi=theta)
        s=np.zeros([1,1,size[2],1,3])
        s[0, 0, :, 0, 0] = x
        s[0, 0, :, 0, 1] = y
        s[0, 0, :, 0, 2] = z
        state.upload(s)
        path.append(state)
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