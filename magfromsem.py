import numpy as np
import ast
import magnes
from pathlib import Path
import sys

def from_sem(directory,save_file=None):
    directory=Path(directory)
    file = open(directory.joinpath('values'),'r')
    contents = file.read()
    file.close()
    contents=contents.replace(', ','\n').replace(',','\n').replace('\n\n','\n').replace(':','": "').replace('\n','", "')
    contents='{"'+contents[:-3]+'}'
    dictionary = ast.literal_eval(contents)
    print(dictionary)
    Nz=int(dictionary['amount of directors'])+1
    D=float(dictionary['d = 2PI Thick / q0'].split(' ')[0])
    K2=float(dictionary['K2'].split(' ')[0])
    U=float(dictionary['K2'].split(' ')[0])
    F=U*1
    W_phi=float(dictionary['W_phi'].split(' ')[0])
    W_theta=float(dictionary['W_theta'].split(' ')[0])
    Kx0=W_phi*1
    Kz0=W_theta*1


    file = np.genfromtxt(directory.joinpath('transition')).T
    n=file[0]
    assert len(n)==Nz
    theta=file[1]
    phi=file[2]
    z=np.cos(theta)
    x=np.sin(theta)*np.cos(phi)
    y=np.sin(theta)*np.sin(phi)
    theta_F=file[3]
    phi_F=file[4]
    s=np.zeros([1,1,1,Nz,1,3])
    s[0,0,0,:,0,0]=x
    s[0,0,0,:,0,1]=y
    s[0,0,0,:,0,2]=z

    size=[1,1,Nz]
    primitives = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
    representatives = [(0., 0., 0.)]
    bc=[magnes.BC.PERIODIC,magnes.BC.PERIODIC,magnes.BC.FREE]
    system = magnes.System(primitives, representatives, size, bc)
    origin = magnes.Vertex(cell=[0, 0, 0])
    J=1.
    D = D
    system.add(magnes.Exchange(origin, magnes.Vertex([1, 0, 0]), J, [D, 0., 0.]))
    system.add(magnes.Exchange(origin, magnes.Vertex([0, 1, 0]), J, [0., D, 0.]))
    system.add(magnes.Exchange(origin, magnes.Vertex([0, 0, 1]), J, [0., 0., D]))
    system.add(magnes.Anisotropy(np.power(D, 2) * F))
    Kx = np.zeros(Nz)
    Kx[0] = Kx0
    Kx[-1] = Kx0
    Kx = np.power(D, 2)*Kx.reshape(1, 1, Nz, 1)
    system.add(magnes.Anisotropy(Kx, axis=[1, 0, 0]))
    Kz = np.zeros(Nz)
    Kz[0] = Kz0
    Kz[-1] = Kz0
    Kz = np.power(D, 2) * Kz.reshape(1, 1, Nz, 1)
    system.add(magnes.Anisotropy(Kz, axis=[0, 0, 1]))
    print(system)
    state=system.field3D()
    state.upload(list(s)[0])
    state.satisfy_constrains()
    s=state.download()

    if save_file is None:
        save_file = directory.stem+'.npz'

    container = magnes.io.container(str(directory.joinpath(save_file)))
    container.store_system(system)
    container["PATH"] = np.array([s])
    container.save_npz(str(directory.joinpath(save_file)))

if __name__ == "__main__":
    directory=sys.argv[1]
    name = None if len(sys.argv)<2 else sys.argv[2]
    from_sem(directory=directory,save_file=name)