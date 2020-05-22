import numpy as np
import sys
import os
import shutil
import glob

import magnes
import magnes.graphics

import matplotlib.pyplot as plt
import map_file_manager as mfm
import map_info
import scipy.interpolate
from pathlib import Path

def change_shape(state,xsize):
    size0 = list(state.shape)
    ini = np.zeros([xsize, size0[1], size0[2], 1, 3])
    if size0[2]==1:
        sx = state[:, 0, 0, 0, 0]
        sy = state[:, 0, 0, 0, 1]
        sz = state[:, 0, 0, 0, 2]
        interp_x = scipy.interpolate.interp1d(np.linspace(0, 1, size0[0], endpoint=True), sx)
        interp_y = scipy.interpolate.interp1d(np.linspace(0, 1, size0[0], endpoint=True), sy)
        interp_z = scipy.interpolate.interp1d(np.linspace(0, 1, size0[0], endpoint=True), sz)
        for x in range(xsize):
            for y in range(size0[1]):
                ini[x, y, 0, 0, 0] = interp_x(x /(xsize-1))
                ini[x, y, 0, 0, 1] = interp_y(x /(xsize-1))
                ini[x, y, 0, 0, 2] = interp_z(x /(xsize-1))
    else:
        sx = state[:, 0, :, 0, 0]
        sy = state[:, 0, :, 0, 1]
        sz = state[:, 0, :, 0, 2]
        interp_x = scipy.interpolate.interp2d(np.linspace(0, 1, size0[0], endpoint=True),
                                              np.linspace(0, 1, size0[2], endpoint=True), sx.T)
        interp_y = scipy.interpolate.interp2d(np.linspace(0, 1, size0[0], endpoint=True),
                                              np.linspace(0, 1, size0[2], endpoint=True), sy.T)
        interp_z = scipy.interpolate.interp2d(np.linspace(0, 1, size0[0], endpoint=True),
                                              np.linspace(0, 1, size0[2], endpoint=True), sz.T)
        for x in range(xsize):
            for y in range(size0[1]):
                for z in range(size0[2]):
                    ini[x, y, z, 0, 0] = interp_x(x /(xsize-1), z / (size0[2]-1))
                    ini[x, y, z, 0, 1] = interp_y(x /(xsize-1), z / (size0[2]-1))
                    ini[x, y, z, 0, 2] = interp_z(x /(xsize-1), z / (size0[2]-1))
    return ini


def one_minimize(directory,load_file,save_file,Kbulk,Ksurf,reshape=None):
    directory=Path(directory)
    load_file=Path(load_file)
    save_file=Path(save_file)
    container = magnes.io.load(os.path.join(directory,load_file))
    ini = container["STATE"]
    if reshape:
        ini=change_shape(ini,reshape)
    size=list(ini.shape)[:3]

    Nz=ini.shape[2]
    primitives = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
    representatives = [(0., 0., 0.)]
    bc=[magnes.BC.PERIODIC,magnes.BC.PERIODIC,magnes.BC.FREE]

    J=1.
    D=np.tan(np.pi/10)

    system = magnes.System(primitives, representatives, size, bc)
    origin = magnes.Vertex(cell=[0, 0, 0])
    system.add(magnes.Exchange(origin, magnes.Vertex([1, 0, 0]), J, [D, 0., 0.]))
    system.add(magnes.Exchange(origin, magnes.Vertex([0, 1, 0]), J, [0., D, 0.]))
    system.add(magnes.Exchange(origin, magnes.Vertex([0, 0, 1]), J, [0., 0., D]))
    K = Kbulk * np.ones(Nz)
    K[0] = Kbulk + Ksurf
    K[-1] = Kbulk + Ksurf
    K = K.reshape(1, 1, Nz, 1)
    system.add(magnes.Anisotropy(K))
    state = system.field3D()
    state.upload(ini)
    state.satisfy_constrains()
    #fig, _, _ = magnes.graphics.plot_field3D(system, state, slice2D='xz', sliceN=int(system.size[1] / 2))
    #plt.show()
    maxtime = 1000
    alpha = 0.1
    precision = 2.5e-5
    catcher = magnes.EveryNthCatcher(1000)

    reporters = []#[magnes.TextStateReporter()]#,magnes.graphics.VectorStateReporter3D(slice2D='xz',sliceN=0)]

    '''mask = np.ones(size+[1])
    mask[4::int(size[0]/20),:,2:-2,0]=0
    freeze_mask = system.field1D().upload(mask)
    minimizer = magnes.StateGDwM(system,freeze=freeze_mask, reference=None, stepsize=alpha, maxiter=None,
                                maxtime=maxtime, precision=precision, catcher=catcher,
                                reporter=magnes.MultiReporter(reporters))#,speedup=1)

    minimizer.optimize(state)

    mask = np.ones(size + [1])
    mask[:, :, int(size[2]/2), 0] = 0
    freeze_mask = system.field1D().upload(mask)
    minimizer = magnes.StateGDwM(system, freeze=freeze_mask, reference=None, stepsize=alpha, maxiter=None,
                                 maxtime=maxtime, precision=precision, catcher=catcher,
                                 reporter=magnes.MultiReporter(reporters))  # ,speedup=1)

    minimizer.optimize(state)'''
    minimizer=magnes.StateNCG(system, reference=None, stepsize=alpha, maxiter=None, maxtime=200, precision=precision,
                              reporter=magnes.MultiReporter(reporters), catcher=catcher)
    minimizer.optimize(state)
    s = state.download()
    print(f'{np.mean(np.abs(s[:,:,:,0,0])) = }')
    container = magnes.io.container()
    container.store_system(system)
    container["STATE"] = s
    #container["ENERGY"] =state.energy_contributions_sum()['total']
    container.save(os.path.join(directory,save_file))
    return state.energy_contributions_sum()['total']/(s.shape[0]*s.shape[1]*s.shape[2]),np.mean(np.abs(s[:,:,2:-2,0,0]))


def two_minimize(directory,load_file,save_file,Kbulk,Ksurf,reshape=None):
    container = magnes.io.load(directory+load_file)
    ini = container["STATE"]
    if reshape:
        ini=change_shape(ini,reshape)
    size=list(ini.shape)[:3]

    Nz=ini.shape[2]
    primitives = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
    representatives = [(0., 0., 0.)]
    bc=[magnes.BC.PERIODIC,magnes.BC.PERIODIC,magnes.BC.FREE]

    J=1.
    D=np.tan(np.pi/10)

    system = magnes.System(primitives, representatives, size, bc)
    origin = magnes.Vertex(cell=[0, 0, 0])
    system.add(magnes.Exchange(origin, magnes.Vertex([1, 0, 0]), J, [D, 0., 0.]))
    system.add(magnes.Exchange(origin, magnes.Vertex([0, 1, 0]), J, [0., D, 0.]))
    system.add(magnes.Exchange(origin, magnes.Vertex([0, 0, 1]), J, [0., 0., D]))
    K = Kbulk * np.ones(Nz)
    K[0] = Kbulk + Ksurf
    K[-1] = Kbulk + Ksurf
    K = K.reshape(1, 1, Nz, 1)
    system.add(magnes.Anisotropy(K))
    state = system.field3D()
    state.upload(ini)
    state.satisfy_constrains()
    #fig, _, _ = magnes.graphics.plot_field3D(system, state, slice2D='xz', sliceN=int(system.size[1] / 2))
    #plt.show()
    maxtime = 1000
    alpha = 0.1
    precision = 5e-5
    catcher = magnes.EveryNthCatcher(1000)

    reporters = [magnes.TextStateReporter()]#,magnes.graphics.VectorStateReporter3D(slice2D='xz',sliceN=0)]

    '''mask = np.ones(size+[1])
    mask[4::int(size[0]/20),:,2:-2,0]=0
    freeze_mask = system.field1D().upload(mask)
    minimizer = magnes.StateGDwM(system,freeze=freeze_mask, reference=None, stepsize=alpha, maxiter=None,
                                maxtime=maxtime, precision=precision, catcher=catcher,
                                reporter=magnes.MultiReporter(reporters))#,speedup=1)

    minimizer.optimize(state)

    mask = np.ones(size + [1])
    mask[:, :, int(size[2]/2), 0] = 0
    freeze_mask = system.field1D().upload(mask)
    minimizer = magnes.StateGDwM(system, freeze=freeze_mask, reference=None, stepsize=alpha, maxiter=None,
                                 maxtime=maxtime, precision=precision, catcher=catcher,
                                 reporter=magnes.MultiReporter(reporters))  # ,speedup=1)

    minimizer.optimize(state)'''
    minimizer=magnes.StateNCG(system, reference=None, stepsize=alpha, maxiter=None, maxtime=200, precision=precision,
                              reporter=magnes.MultiReporter(reporters), catcher=catcher)
    minimizer.optimize(state)
    s = state.download()
    print(f'{np.mean(np.abs(s[:,:,:,0,0])) = }')
    container = magnes.io.container()
    container.store_system(system)
    container["STATE"] = s
    #container["ENERGY"] =state.energy_contributions_sum()['total']
    container.save(directory+save_file)
    return state.energy_contributions_sum()['total']/(s.shape[0]*s.shape[1]*s.shape[2]),np.mean(np.abs(s[:,:,2:-2,0,0]))


def minimize(directory,load_file,save_file,Kbulk,Ksurf,reshape=None,z0=0.25):
    ans=one_minimize(directory,load_file,save_file,Kbulk,Ksurf,reshape)
    if ans[1]<z0:
        return ans
    print('first')
    one_minimize(directory,load_file,save_file,0.,0.,reshape)
    ans= one_minimize(directory, save_file, save_file, Kbulk, Ksurf)
    if ans[1]<z0:
        print('success')
        return ans
    return ans

def parabola(x, a, b, c):
    return a*x**2 + b*x + c

def min_period(energy):
    nnan_energy=energy[np.invert(np.isnan(energy[:,1]))]
    n = np.nanargmin(nnan_energy[:, 1])
    apr = np.abs(np.array(range(nnan_energy.shape[0])) - n) < 20
    nnan_energy=nnan_energy[apr]
    if nnan_energy.shape[0]==1:
        n= nnan_energy[0,0]
        pb = np.array([0.,0.,nnan_energy[0,1]])
    elif nnan_energy.shape[0] == 2:
        dx=nnan_energy[1,0]-nnan_energy[0,0]
        dy = nnan_energy[1, 1] - nnan_energy[0, 1]
        n=nnan_energy[np.argmin(nnan_energy[:, 1]),0]
        pb=np.array([0.,dy/dx ,nnan_energy[0, 1]- dy/dx*nnan_energy[0, 0]])
    else:
        try:
            pb,_=scipy.optimize.curve_fit(parabola, nnan_energy[:, 0], nnan_energy[:, 1])
            n= -pb[1] / (2 * pb[0])
            if pb[0] < 0:
                if n> energy[0,0]:
                    n = energy[0,0]
                else:
                    n= energy[-1,0]
        except:
            n=[energy[np.nanargmin(energy[:,1]),0]]
            pb=np.array([0.,0.,nnan_energy.shape[0,1]])
    print(f'{n = }')
    print(f'{pb = }')
    return n,pb

def period_plot(energy,Kbulk, Ksurf,pb=None,wrong_energy=[]):
    energy=np.array(energy)
    print(f'{n = }\t{energy[0,0] = }\t{energy[-1, 0] = }')
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('period')
    ax1.set_ylabel('energy', color='r')
    ax1.plot(energy[:, 0], energy[:, 1], 'r.')
#    if len(wrong_energy)>0:
#        print(f'{np.array(wrong_energy) = }')
#        ax1.plot(np.array(wrong_energy)[:, 0], np.array(wrong_energy)[:, 1], 'k3')
    if pb is not None:
        spline_period = np.linspace(energy[:, 0].min(), energy[:, 0].max(), 50)
        ax1.plot(spline_period, parabola(spline_period, *pb), 'm--')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('$\\langle S_x \\rangle$', color='b')  # we already handled the x-label with ax1
    ax2.plot(energy[:, 0], energy[:, 2], 'b.')
    plt.title(f'{Kbulk = :.2f}, {Ksurf = :.2f}')
    plt.tight_layout()
    plt.show()

def next_point(energy,n0,N,z0):
    energy=np.array(sorted(np.array(energy).tolist(), key=lambda x: x[0]))
    wl=energy[:,2]>z0
    wrong_energy=energy[wl]
    energy[wl,1]=np.nan
    nnan_energy = energy[np.invert(np.isnan(energy[:, 1]))]
    n,pb=min_period(energy)
    print(f'{n = }')
    if pb[0]>0 and n<energy[-min(n0,len(energy)-1),0] and n>energy[min(n0,len(energy)-1),0] and (len(nnan_energy)>=5 or len(energy)>20):
        period = None
        ref = None
    elif pb[0]>0:
        if np.abs(n-energy[:,0].min())<np.abs(n-energy[:,0].max()):
            period=np.round(np.nanmin(energy[:,0])-1/N,5)
            ref=np.nanmin(nnan_energy[:,0])
        else:
            period=np.round(np.nanmax(energy[:,0])+1/N,5)
            ref = np.nanmax(nnan_energy[:, 0])
    else:
        print(np.abs(nnan_energy[np.nanargmin(nnan_energy[:,1]),0]-energy[:,0].min()))
        print(np.abs(nnan_energy[np.nanargmin(nnan_energy[:, 1]), 0] - energy[:, 0].max()))
        if np.abs(nnan_energy[np.nanargmin(nnan_energy[:,1]),0]-energy[:,0].min())>\
            np.abs(nnan_energy[np.nanargmin(nnan_energy[:,1]),0]-energy[:,0].max()):
            period=np.round(np.nanmax(energy[:,0])+1/N,5)
            ref=np.nanmax(nnan_energy[:,0])
        else:
            period=np.round(np.nanmin(energy[:,0])-1/N,5)
            ref = np.nanmin(nnan_energy[:, 0])
    if period is not None and (period<2 or period>200):
        period = None
        ref = None
    print(f'{n = }\t{energy[0,0] = }\t{energy[-1, 0] = }')
    return energy.tolist(),pb,n,period,ref,nnan_energy, wrong_energy

if __name__ == "__main__":
    initial = Path('/home/ivan/LC_SK/initials/matspx10_1_20.npz')
    directory = Path('/home/ivan/LC_SK/spx/5/')
    state_name = 'matspx'

    if not os.path.exists(directory):
        os.makedirs(directory)

    ###############
    N=10
    n0 = 5
    z0 = 0.25
    ###########

    if len([f for f in os.listdir(directory) if Path(f).suffix=='.npz'])==0:
        energy = []
        Kbulk= 0.0
        Ksurf = 0.
        container = magnes.io.load(str(initial))
        ini = container["STATE"]
        period = ini.shape[0]/N
        print(f'{period = }')
        energy.append([period, *minimize(directory=Path(''),
                                         load_file=initial,
                                         save_file=os.path.join(str(directory),state_name +
                                                                '_{:.5f}_{:.5f}_{:.5f}.npz'.format(Kbulk, Ksurf, period)),
                                         Kbulk=Kbulk * (np.tan(np.pi / 10)) ** 2,
                                         Ksurf=Ksurf * (np.tan(np.pi / 10)) ** 2,z0=z0)])
        energy=np.array(energy)
        energy, pb,n, period, ref,nnan_energy, wrong_energy=next_point(energy,n0,N,z0)
        while period:
            #period_plot(energy, Kbulk, Ksurf, pb, wrong_energy)
            energy.append([period, *minimize(directory=directory,
                                             load_file=state_name + '_{:.5f}_{:.5f}_{:.5f}.npz'.format(Kbulk, Ksurf,ref),
                                             save_file=state_name + '_{:.5f}_{:.5f}_{:.5f}.npz'.format(Kbulk, Ksurf,period),
                                             Kbulk=Kbulk * (np.tan(np.pi / 10)) ** 2,
                                             Ksurf=Ksurf * (np.tan(np.pi / 10)) ** 2, z0=z0,reshape=int(period*N))])
            energy, pb,n, period, ref,nnan_energy, wrong_energy = next_point(energy,n0,N,z0)
        #period_plot(energy, Kbulk, Ksurf, pb)

    Klist,Kaxis = mfm.file_manager(directory,
                             params={'double': False,
                                     'add': [np.round(np.linspace(0.0, -0.5,11), decimals=5).tolist(),
                                            np.round(np.linspace(5,5, 1), decimals=5).tolist()]
                                     },dimension=2
                             )
    if len(Klist>0):
        if len(Kaxis[0])>1 and len(Kaxis[1])>1:
            scale=np.array([np.abs(Klist)[:, 1].max(), np.abs(Klist)[:, 0].max()])
        else:
            scale=1.
        Klist=Klist[np.array(sorted(list(enumerate(Klist)),
                              key=lambda x: np.linalg.norm(x[1] * scale)))[:, 0].tolist()]
    for idx,Kv in enumerate(Klist,start=1):
        complete,_ = mfm.content(directory)
        Kbulk = Kv[0]
        Ksurf = Kv[1]
        print(f'\n\n\n\n{Kbulk = }\t{Ksurf = }\n\n\n\n')
        bulk_idx=np.where(Kaxis[0]==Kbulk)[0][0]
        surf_idx = np.where(Kaxis[1] == Ksurf)[0][0]
        if bulk_idx<len(Kaxis[0])-1:
            initial = [i for i in complete if i[0]==Kaxis[0][bulk_idx+1] and i[1]==Kaxis[1][surf_idx]]
        else:
            initial = [i for i in complete if i[0] == Kaxis[0][bulk_idx] and i[1] == Kaxis[1][surf_idx-1]]
        initial=sorted(initial,key = lambda x : x[2] )

        energy=[]
        for i in initial:
            period=i[2]
            energy.append([period,*minimize(directory=directory,
                                            load_file=state_name+'_{:.5f}_{:.5f}_{:.5f}.npz'.format(i[0],i[1],i[2]),
                                            save_file=state_name+'_{:.5f}_{:.5f}_{:.5f}.npz'.format(Kbulk,Ksurf,period),
                              Kbulk=Kbulk*(np.tan(np.pi/10))**2,Ksurf=Ksurf*(np.tan(np.pi/10))**2,z0=z0)])
        energy, pb, n, period, ref, nnan_energy, wrong_energy = next_point(energy, n0, N, z0)
        #period_plot(energy,Kbulk,Ksurf,pb,wrong_energy)
        while period:
            #period_plot(energy, Kbulk, Ksurf, pb, wrong_energy)
            energy.append([period, *minimize(directory=directory,
                                             load_file=state_name + '_{:.5f}_{:.5f}_{:.5f}.npz'.format(Kbulk, Ksurf,ref),
                                             save_file=state_name + '_{:.5f}_{:.5f}_{:.5f}.npz'.format(Kbulk, Ksurf,period),
                                             Kbulk=Kbulk * (np.tan(np.pi / 10)) ** 2,
                                             Ksurf=Ksurf * (np.tan(np.pi / 10)) ** 2, z0=z0,reshape=int(period*N))])
            energy, pb,n, period, ref,nnan_energy, wrong_energy = next_point(energy,n0,N,z0)
        #period_plot(energy, Kbulk, Ksurf, pb)
        energy=np.array(energy)
        p0=energy[np.nanargmin(energy[:,1]),0]
        print(f'{p0 = }')
        nnan_energy=energy[np.invert(np.isnan(energy[:,1]))]
        delete_list=energy[np.isnan(energy[:,1])].tolist()
        n=np.nanargmin(nnan_energy[:,1])
        for idx,i in enumerate(nnan_energy):
            if np.abs(idx-n)>n0+2:
                delete_list.append(i)
        for p,e,z in delete_list:
            os.remove(os.path.join(str(directory),state_name + '_{:.5f}_{:.5f}_{:.5f}.npz'.format(Kbulk, Ksurf,p)))

    map_info.map_info(directory)