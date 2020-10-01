import numpy as np
import sys

import os

import magnes
import magnes.graphics

import matplotlib.pyplot as plt
import map_file_manager as mfm
import map_info
import map_color
from pathlib import Path

def change_anisotropy(file,save,K_factor):
    if os.path.isdir(file):
        for f in os.listdir(file):
            container = magnes.io.load(file+f)
            ini = container["STATE"]
            size = list(ini.shape[0:3])

            Nz = ini.shape[2]
            primitives = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
            representatives = [(0., 0., 0.)]
            bc = [magnes.BC.PERIODIC, magnes.BC.PERIODIC, magnes.BC.FREE]

            J = 1.
            D = np.tan(np.pi / 10)

            system = magnes.System(primitives, representatives, size, bc)
            origin = magnes.Vertex(cell=[0, 0, 0])
            system.add(magnes.Exchange(origin, magnes.Vertex([1, 0, 0]), J, [D, 0., 0.]))
            system.add(magnes.Exchange(origin, magnes.Vertex([0, 1, 0]), J, [0., D, 0.]))
            system.add(magnes.Exchange(origin, magnes.Vertex([0, 0, 1]), J, [0., 0., D]))
            K = K_bulk * np.ones(Nz)
            K[0] = K_bulk + K_surf
            K[-1] = K_bulk + K_surf
            K = K.reshape(1, 1, Nz, 1)
            system.add(magnes.Anisotropy(K))
            print(system)
            state = system.field3D()
            state.upload(ini)
            state.satisfy_constrains()
            s = state.download()
            container = magnes.io.container()
            container.store_system(system)
            container["STATE"] = s
            container.save(file+f)
    elif os.path.isfile(file):
        container = magnes.io.load(file)
        path = container["PATH"]
        old_system=container.extract_system()
        H=old_system.field
        exchange=old_system.exchange
        anisotropy=old_system.anisotropy
        size = old_system.size

        primitives = old_system.primitives
        representatives =old_system.representatives
        bc = old_system.bc

        system = magnes.System(primitives, representatives, size, bc)
        origin = magnes.Vertex(cell=[0, 0, 0])
        for ex in exchange:
            system.add(ex)
        system.set_external_field(H)
        for ani in anisotropy:
            system.add(magnes.Anisotropy(axis=ani.axis,strength=ani.strength*K_factor))
        print(system)
        container = magnes.io.container(save)
        container.store_system(system)
        container["PATH"] = path
        container.save(save)
    else:
        print("It is a special file (socket, FIFO, device file)")



def change_everything(file,save,K_bulk,K_surf):
    if os.path.isdir(file):
        for f in os.listdir(file):
            container = magnes.io.load(file+f)
            ini = container["STATE"]
            size = list(ini.shape[0:3])

            Nz = ini.shape[2]
            primitives = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
            representatives = [(0., 0., 0.)]
            bc = [magnes.BC.PERIODIC, magnes.BC.PERIODIC, magnes.BC.FREE]

            J = 1.
            D = np.tan(np.pi / 10)

            system = magnes.System(primitives, representatives, size, bc)
            origin = magnes.Vertex(cell=[0, 0, 0])
            system.add(magnes.Exchange(origin, magnes.Vertex([1, 0, 0]), J, [D, 0., 0.]))
            system.add(magnes.Exchange(origin, magnes.Vertex([0, 1, 0]), J, [0., D, 0.]))
            system.add(magnes.Exchange(origin, magnes.Vertex([0, 0, 1]), J, [0., 0., D]))
            K = K_bulk * np.ones(Nz)
            K[0] = K_bulk + K_surf
            K[-1] = K_bulk + K_surf
            K = K.reshape(1, 1, Nz, 1)
            system.add(magnes.Anisotropy(K))
            print(system)
            state = system.field3D()
            state.upload(ini)
            state.satisfy_constrains()
            s = state.download()
            container = magnes.io.container()
            container.store_system(system)
            container["STATE"] = s
            container.save(file+f)
    elif os.path.isfile(file):
        container = magnes.io.load(file)
        ini = container["STATE"]
        size = list(ini.shape[0:3])

        Nz = ini.shape[2]
        primitives = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
        representatives = [(0., 0., 0.)]
        bc = [magnes.BC.PERIODIC, magnes.BC.PERIODIC, magnes.BC.FREE]

        J = 1.
        D = np.tan(np.pi / 10)

        system = magnes.System(primitives, representatives, size, bc)
        origin = magnes.Vertex(cell=[0, 0, 0])
        system.add(magnes.Exchange(origin, magnes.Vertex([1, 0, 0]), J, [D, 0., 0.]))
        system.add(magnes.Exchange(origin, magnes.Vertex([0, 1, 0]), J, [0., D, 0.]))
        system.add(magnes.Exchange(origin, magnes.Vertex([0, 0, 1]), J, [0., 0., D]))
        K = K_bulk * np.ones(Nz)
        K[0] = K_bulk + K_surf
        K[-1] = K_bulk + K_surf
        K = K.reshape(1, 1, Nz, 1)
        system.add(magnes.Anisotropy(K))
        print(system)
        state = system.field3D()
        state.upload(ini)
        state.satisfy_constrains()
        s = state.download()
        container = magnes.io.container()
        container.store_system(system)
        container["STATE"] = s
        container.save(save)
    else:
        print("It is a special file (socket, FIFO, device file)")


if __name__ == "__main__":
    change_anisotropy(sys.argv[1],sys.argv[2],float(sys.argv[3]))
