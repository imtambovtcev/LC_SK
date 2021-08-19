import numpy as np
import sys
import os
import shutil
import glob

import magnes
import magnes.graphics
import utilities

import matplotlib.pyplot as plt
import map_file_manager as mfm
import map_info
import scipy.interpolate
from pathlib import Path

from termcolor import colored

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def change_x_shape(state, xsize):
    size0 = list(state.shape)
    ini = np.zeros([xsize, size0[1], size0[2], 1, 3])
    # 2d
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
    #3d
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
                    #print(f'{xsize-1 = }{size0[2]-1 = }')
                    ini[x, y, z, 0, 0] = interp_x(x /(xsize-1), z / (size0[2]-1))
                    ini[x, y, z, 0, 1] = interp_y(x /(xsize-1), z / (size0[2]-1))
                    ini[x, y, z, 0, 2] = interp_z(x /(xsize-1), z / (size0[2]-1))
    return ini

def change_y_shape(state, ysize):
    size0 = list(state.shape)
    ini = np.zeros([size0[0], ysize, size0[1], 1, 3])

    sx = state[:, 0, :, 0, 0]
    sy = state[:, 0, :, 0, 1]
    sz = state[:, 0, :, 0, 2]
    interp_x = scipy.interpolate.interp2d(np.linspace(0, 1, size0[0], endpoint=True),
                                          np.linspace(0, 1, size0[2], endpoint=True), sx.T)
    interp_y = scipy.interpolate.interp2d(np.linspace(0, 1, size0[0], endpoint=True),
                                          np.linspace(0, 1, size0[2], endpoint=True), sy.T)
    interp_z = scipy.interpolate.interp2d(np.linspace(0, 1, size0[0], endpoint=True),
                                          np.linspace(0, 1, size0[2], endpoint=True), sz.T)
    for x in range(size0[0]):
        for y in range(ysize):
            for z in range(size0[2]):
                #print(f'{xsize-1 = }{size0[2]-1 = })
                ini[x, y, z, 0, 0] = interp_x(x/(size0[0]-1), y / (ysize-1))
                ini[x, y, z, 0, 1] = interp_y(x /(size0[0]-1),y / (ysize-1))
                ini[x, y, z, 0, 2] = interp_z(x /(size0[0]-1),y / (ysize-1))
    return ini

def change_z_shape(state, zsize):
    size0 = list(state.shape)
    ini = np.zeros([size0[0], size0[1], zsize, 1, 3])

    sx = state[:, 0, :, 0, 0]
    sy = state[:, 0, :, 0, 1]
    sz = state[:, 0, :, 0, 2]
    interp_x = scipy.interpolate.interp2d(np.linspace(0, 1, size0[0], endpoint=True),
                                          np.linspace(0, 1, size0[2], endpoint=True), sx.T)
    interp_y = scipy.interpolate.interp2d(np.linspace(0, 1, size0[0], endpoint=True),
                                          np.linspace(0, 1, size0[2], endpoint=True), sy.T)
    interp_z = scipy.interpolate.interp2d(np.linspace(0, 1, size0[0], endpoint=True),
                                          np.linspace(0, 1, size0[2], endpoint=True), sz.T)
    for x in range(size0[0]):
        for y in range(size0[1]):
            for z in range(zsize):
                #print(f'{xsize-1 = }{size0[2]-1 = }')
                ini[x, y, z, 0, 0] = interp_x(x/(size0[0]-1), z / (zsize-1))
                ini[x, y, z, 0, 1] = interp_y(x /(size0[0]-1),z / (zsize-1))
                ini[x, y, z, 0, 2] = interp_z(x /(size0[0]-1),z / (zsize-1))
    return ini

def minimize(ini,J=1.,D=np.tan(np.pi/10),Kbulk=0.,Ksurf=0.,
             primitives = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)],representatives = [(0., 0., 0.)],
             bc = [magnes.BC.PERIODIC, magnes.BC.PERIODIC, magnes.BC.FREE],
             maxtime=None,maxiter = None, alpha = 0.5, precision = 1e-7,catcherN=1000):
    size = list(ini.shape)[:3]
    Nz = ini.shape[2]
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
    # fig, _, _ = magnes.graphics.plot_field3D(system, state, slice2D='xz', sliceN=int(system.size[1] / 2))
    # plt.show()
    catcher = magnes.EveryNthCatcher(catcherN)

    reporters = []  # [magnes.TextStateReporter()]#,magnes.graphics.VectorStateReporter3D(slice2D='xz',sliceN=0)]
    minimizer = magnes.StateNCG(system, reference=None, stepsize=alpha, maxiter=maxiter, maxtime=maxtime, precision=precision,
                                reporter=magnes.MultiReporter(reporters), catcher=catcher)
    blockPrint()
    minimizer.optimize(state)
    enablePrint()
    return system,state.download(),state.energy_contributions_sum()['total']


def minimize_from_file(directory,load_file,save_file,J=1.,D=np.tan(np.pi/10),Kbulk=0.,Ksurf=0.,reshape=None,
                       z_max_proj=None, precision = 1e-7, maxiter = None, boundary=['P','P','F']):
    directory=Path(directory)
    load_file=Path(load_file)
    save_file=Path(save_file)
    container = magnes.io.load(str(directory.joinpath(load_file)))
    if 'STATE' in container:
        ini = container["STATE"]
    else:
        ini = list(container["PATH"])[0]
    if reshape:
        ini=change_x_shape(ini, reshape)
    bc = [magnes.BC.PERIODIC if i == 'P' else magnes.BC.FREE for i in boundary]
    system,s,energy=minimize(ini,J=J,D=D,Kbulk=Kbulk,Ksurf=Ksurf, precision = precision,maxiter = maxiter, bc=bc)
    #print(f'{np.mean(np.abs(s[:,:,:,0,0])) = }')
    container = magnes.io.container(str(directory.joinpath(save_file)))
    container.store_system(system)
    container["PATH"] = np.array([s])
    container["ENERGY"] = energy
    container.save_npz(str(directory.joinpath(save_file)))
    if z_max_proj is not None:
        zp=np.mean(np.abs(s[:,:,2:-2,0,0]))
        return energy / (s.shape[0] * s.shape[1] * s.shape[2]),zp
    else:
        return energy/(s.shape[0]*s.shape[1]*s.shape[2])

def minimize_from_state(directory,load_state,save_file,J=1.,D=np.tan(np.pi/10),Kbulk=0.,Ksurf=0.,reshape=None,z_max_proj=None, precision = 1e-7, boundary=['P','P','F']):
    ini = load_state
    if reshape:
        ini=change_x_shape(ini, reshape)
    bc=[magnes.BC.PERIODIC if i == 'P' else magnes.BC.FREE for i in boundary]
    system,s,energy=minimize(ini,J=J,D=D,Kbulk=Kbulk,Ksurf=Ksurf, precision = precision, bc=bc)
    print(f'{np.mean(np.abs(s[:,:,:,0,0])) = }')
    container = magnes.io.container(str(directory.joinpath(save_file)))
    container.store_system(system)
    container["PATH"] = np.array([s])
    container["ENERGY"] = energy
    container.save_npz(str(directory.joinpath(save_file)))
    if z_max_proj is not None:
        zp=np.mean(np.abs(s[:,:,2:-2,0,0]))
        return energy / (s.shape[0] * s.shape[1] * s.shape[2]),zp
    else:
        return energy/(s.shape[0]*s.shape[1]*s.shape[2])

def parabola(x, a, b, c):
    return a*x**2 + b*x + c

def min_period(energy):
    nnan_energy=energy[np.invert(np.isnan(energy[:,1]))]
    if len(nnan_energy[:, 1])>0:
        n = np.nanargmin(nnan_energy[:, 1])
        apr = np.abs(np.array(range(nnan_energy.shape[0])) - n) < 20
        nnan_energy = nnan_energy[apr]
        if nnan_energy.shape[0] == 1:
            n = nnan_energy[0, 0]
            pb = np.array([0., 0., nnan_energy[0, 1]])
        elif nnan_energy.shape[0] == 2:
            dx = nnan_energy[1, 0] - nnan_energy[0, 0]
            dy = nnan_energy[1, 1] - nnan_energy[0, 1]
            n = nnan_energy[np.argmin(nnan_energy[:, 1]), 0]
            pb = np.array([0., dy / dx, nnan_energy[0, 1] - dy / dx * nnan_energy[0, 0]])
        else:
            try:
                pb, _ = scipy.optimize.curve_fit(parabola, nnan_energy[:, 0], nnan_energy[:, 1])
                n = -pb[1] / (2 * pb[0])
                if pb[0] < 0:
                    if n > energy[0, 0]:
                        n = energy[0, 0]
                    else:
                        n = energy[-1, 0]
            except:
                n = energy[np.nanargmin(energy[:, 1]), 0]
                pb = np.array([0., 0., np.nanmin(energy[:, 1])])
    else:
        n = np.infty
        pb=np.array([0.,0.,0.])

    print(f'{n = }')
    print(f'{pb = }')
    return n,pb

def gaps(points, period_N):
    r=int(np.ceil(np.log10(period_N))+3)
    points_r=np.round(points,r)
    all=list(np.round(np.array(range(int(points_r.min()*period_N),int(points_r.max()*period_N)))/period_N,r))
    points_r=points_r.tolist()
    return np.array([i for i in all if not(i in points_r)])

def next_point(energy,max_steps_from_minimum,period_N,z_max_proj,max_period=np.infty):
    # sort energy by period
    print(f'{energy = }')
    energy=np.array(sorted(np.array(energy).tolist(), key=lambda x: x[0]))
    # get unsutable points
    wl=energy[:,2]>z_max_proj
    wrong_energy=energy[wl]
    energy[wl,1]=np.nan
    nnan_energy = energy[np.invert(np.isnan(energy[:, 1]))]
    n,pb=min_period(energy)
    print(f'{n = }')
    gap = gaps(energy[:, 0], period_N=period_N)
    if len(gap) > 0:
        period = gap[0]
        try:
            ref = nnan_energy[np.argmin(np.abs(nnan_energy[:, 0] - period)), 0]
        except:
            ref = energy[0, 0]
        print('gap found')
    else:
        if len(nnan_energy) == 0 and len(energy)<10:
            period = np.round(np.nanmax(energy[:, 0]) + 1 / period_N, 5)
            ref = np.nanmax(energy[:, 0])
        elif len(energy) > len(nnan_energy)*10:
            period = None
            ref = None
        else:
                right = True
                left = True

                if np.nanmin(energy[:,0])<=2 or n-np.nanmin(nnan_energy[:,0])>max_steps_from_minimum/period_N or \
                        np.nanmin(nnan_energy[:,0])-np.nanmin(energy[:,0])>5*max_steps_from_minimum/period_N:
                    left = False
                if np.nanmax(energy[:,0])>max_period or \
                        np.nanmax(nnan_energy[:,0])-n>max_steps_from_minimum/period_N or \
                        np.nanmax(energy[:,0])-np.nanmax(nnan_energy[:,0])>5*max_steps_from_minimum/period_N:
                    right=False

                if left and right:
                    if pb[0]>0:
                        if n-np.nanmax(nnan_energy[:,0])>np.nanmin(nnan_energy[:,0])-n:
                            right=False
                        else:
                            left=False
                    else:
                        if n-np.nanmax(nnan_energy[:,0])>np.nanmin(nnan_energy[:,0])-n:
                            left=False
                        else:
                            right=False

                if left and not right:
                    period = np.round(np.nanmin(energy[:, 0]) - 1 / period_N, 5)
                    ref = np.nanmin(nnan_energy[:, 0])
                elif not left and right:
                    period = np.round(np.nanmax(energy[:, 0]) + 1 / period_N, 5)
                    ref = np.nanmax(nnan_energy[:, 0])
                elif not(left or right):
                    period = None
                    ref = None
                else:
                    print('lr error')
                    period = None
                    ref = None

    print(f'{n = }\t{period = }\t{ref = }\t{energy[0,0] = }\t{energy[-1, 0] = }')
    return energy.tolist(),pb,n,period,ref,nnan_energy, wrong_energy

def get_reference(K,K_list,reverse=False):
    ref = list(K_list)
    #print(K_list)
    ref=sorted(ref, key= lambda x: np.abs(x[:2]-K).tolist())
    if reverse:
        ref = [i for i in ref if K[0] <= i[0]]
    else:
        ref = [i for i in ref if K[0] >= i[0]]
    ref = [ i for i in ref if i[0] == ref[0][0] and i[1] == ref[0][1] ]
    print(f'{ref = }')
    if len(ref)==0:
        ref = list(K_list)
        ref=sorted(ref, key= lambda x: np.abs(x[:2]-K).tolist())
        ref = [i for i in ref if i[0] == ref[0][0] and i[1] == ref[0][1]]
    print(f'{ref = }')
    return ref

def set_xperiod_point(Kbulk_D,Ksurf_D,initials_set,directory,state_name='matspx',period_N=1,max_steps_from_minimum = 5, z_max_proj = np.infty,max_period = np.infty, precision = 1e-7,D=np.tan(np.pi/10), boundary=['P','P','F']):
    energy = []
    for period,initial in initials_set:
        energy.append([period, *minimize_from_state(directory=Path(''),
                                                   load_state=initial,
                                                   save_file=os.path.join(str(directory), state_name +
                                                                          '_{:.5f}_{:.5f}_{:.5f}.npz'.format(Kbulk_D, Ksurf_D, period)),
                                                   Kbulk=np.power(D, 2) * Kbulk_D,
                                                   Ksurf=np.power(D, 2) * Ksurf_D,
                                                   z_max_proj=z_max_proj, precision = precision, boundary=boundary,D=D)])
    energy= np.array(energy) if len(energy) !=0 else np.array([[0,np.infty,np.infty]])

    energy, pb, n, period, ref, nnan_energy, wrong_energy = next_point(energy, max_steps_from_minimum, period_N,z_max_proj)
    while period:
        # period_plot(energy, Kbulk, Ksurf, pb, wrong_energy)
        energy.append([period, *minimize_from_file(directory=directory,
                                                   load_file=state_name + '_{:.5f}_{:.5f}_{:.5f}.npz'.format(
                                                       Kbulk_D, Ksurf_D, ref),
                                                   save_file=state_name + '_{:.5f}_{:.5f}_{:.5f}.npz'.format(
                                                       Kbulk_D, Ksurf_D, period),
                                                   Kbulk=np.power(D, 2) * Kbulk_D,
                                                   Ksurf=np.power(D, 2) * Ksurf_D,
                                                   reshape=int(period * period_N),
                                                   z_max_proj=z_max_proj, precision = precision,D=D, boundary=boundary)])
        energy, pb, n, period, ref, nnan_energy, wrong_energy = next_point(energy=energy,
                                                                           max_steps_from_minimum=max_steps_from_minimum,
                                                                           period_N=period_N, z_max_proj=z_max_proj,
                                                                           max_period=max_period)

    energy = np.array(energy)
    if np.all(np.isnan(energy[:, 1])):
        p0 = np.nan
    else:
        p0 = energy[np.nanargmin(energy[:, 1]), 0]

    print(f'{p0 = }')
    nnan_energy = energy[np.invert(np.isnan(energy[:, 1]))]
    delete_list = energy[np.isnan(energy[:, 1])].tolist()
    if np.all(np.isnan(energy[:, 1])):
        n=0
    else:
        n = np.nanargmin(nnan_energy[:, 1])
    for idx, i in enumerate(nnan_energy):
        if np.abs(idx - n) > max_steps_from_minimum + 2:
            delete_list.append(i)
    for p, e, z in delete_list:
        os.remove(
            os.path.join(str(directory), state_name + '_{:.5f}_{:.5f}_{:.5f}.npz'.format(Kbulk_D, Ksurf_D, p)))
    return energy

def make_map_from_file_x_minimisation(save_dir,KDbulk_list,KDsurf_list, ref,period_N=1,max_steps_from_minimum = 5,
                                      z_max_proj = np.infty,max_period = np.infty,reverse = True, precision = 1e-7, state_name = 'matspx', D=np.tan(np.pi/10), boundary=['P','P','F']):
    initial = Path(ref)
    directory = Path(save_dir)

    if not os.path.exists(directory):
        os.makedirs(directory)

    Klist, Kaxis = mfm.file_manager(directory,
                                    params={'double': False,
                                            'add': [np.round(KDbulk_list, decimals=5).tolist(),
                                                    np.round(KDsurf_list, decimals=5).tolist()]
                                            }, dimension=2
                                    )

    Klist = np.array(sorted(Klist.tolist(), key=lambda x: [-x[1], x[0]], reverse=reverse))

    print(f'{Klist = }')

    for idx, Kv in enumerate(Klist, start=1):
        complete, _ = mfm.content(directory)
        print(f'{len(complete) = }')
        if len(complete) == 0:
            Kbulk_D = Kv[0]
            Ksurf_D = Kv[1]
            print(colored('initial', 'green'))
            print(colored('K_bulk = {}\tK_surf = {}\n\n'.format(Kbulk_D, Ksurf_D), 'blue'))
            container = magnes.io.load(str(initial))
            if 'STATE' in container:
                ini = container["STATE"]
            else:
                print(f'minimize from 0 image of the path with {container["PATH"].shape = }')
                ini = list(container["PATH"])[0]
            period = ini.shape[0] / period_N

            print(f'{period = }')
            set_xperiod_point(Kbulk_D=Kbulk_D, Ksurf_D=Ksurf_D, initials_set=[[period, ini]], directory=directory,
                              state_name=state_name,
                              period_N=period_N, max_steps_from_minimum=max_steps_from_minimum, z_max_proj=z_max_proj,
                              max_period=max_period, precision=precision,D=D, boundary=boundary)
        else:
            Kbulk_D = Kv[0]
            Ksurf_D = Kv[1]
            print(colored('\n\nK_bulk = {}\tK_surf = {}\n\n'.format(Kbulk_D, Ksurf_D), 'blue'))

            initial = get_reference(Kv, complete, reverse) #Kbulk_D,Ksurf_D,period
            initial = sorted(initial, key=lambda x: x[2])

            initial_set=[]
            for i in initial:
                container = magnes.io.load(str(directory.joinpath(state_name + '_{:.5f}_{:.5f}_{:.5f}.npz'.format(i[0], i[1], i[2]))))
                if 'STATE' in container:
                    ini = container["STATE"]
                else:
                    ini = list(container["PATH"])[0]
                initial_set.append([i[2],ini])

            set_xperiod_point(Kbulk_D=Kbulk_D, Ksurf_D=Ksurf_D, initials_set=initial_set, directory=directory,
                              state_name=state_name,
                              period_N=period_N, max_steps_from_minimum=max_steps_from_minimum, z_max_proj=z_max_proj,
                              max_period=max_period, precision = precision,D=D, boundary=boundary)


def make_map_by_multiplication_x_minimisation(save_dir, ref_dir,initial_period_N=1,period_N=1,max_steps_from_minimum = 5, z_max_proj = np.infty,max_period = np.infty,reverse = True, precision = 1e-7,state_name = 'matspx',J = 1.0,D = np.tan(np.pi / 10)):
    directory = Path(save_dir)

    if not os.path.exists(directory):
        os.makedirs(directory)

    ref,initial_state_name=mfm.content(ref_dir)
    ref_axis=mfm.axis(ref)[:2]
    print(f'{ref = }')
    print(f'{ref_axis = }')
    for Kbulk_D in ref_axis[0]:
        for Ksurf_D in ref_axis[1]:
            min_ini_energy=np.infty
            for i in ref[np.logical_and(ref[:,0]==Kbulk_D,ref[:,1]==Ksurf_D)]:
                container = magnes.io.load(
                    str(ref_dir.joinpath(initial_state_name + '_{:.5f}_{:.5f}_{:.5f}.npz'.format(i[0], i[1], i[2]))))
                system=container.extract_system()
                if 'STATE' in container:
                    ini = container["STATE"]
                else:
                    ini = list(container["PATH"])[0]
                state = system.field3D()
                state.upload(ini)
                if utilities.get_energy(state)[1]< min_ini_energy:
                    min_ini_energy=utilities.get_energy(state)[1]
                    ini0=np.copy(ini)
                    for ct in range(period_N - initial_period_N):
                        ini = np.concatenate([ini, ini0], axis=0)
                    initial_set = [[i[2], ini]]

            set_xperiod_point(Kbulk_D=Kbulk_D, Ksurf_D=Ksurf_D, initials_set=initial_set, directory=directory,
                              state_name=state_name,
                              period_N=period_N, max_steps_from_minimum=max_steps_from_minimum, z_max_proj=z_max_proj,
                              max_period=max_period, precision = precision)

def make_map_from_file(save_dir, KDbulk_list, KDsurf_list, ref, reverse = True, precision = 1e-7, maxiter = None, state_name = 'magnpzfile',J = 1.0,D=np.tan(np.pi/10), boundary=['P','P','F']):
    initial = Path(ref)
    directory = Path(save_dir)

    if not os.path.exists(directory):
        os.makedirs(directory)

    Klist, Kaxis = mfm.file_manager(directory,
                                    params={'double': False,
                                            'add': [np.round(KDbulk_list, decimals=5).tolist(),
                                                    np.round(KDsurf_list, decimals=5).tolist()]
                                            }, dimension=2
                                    )


    Klist = np.array(sorted(Klist.tolist(), key=lambda x: [-x[1], x[0]], reverse=reverse))

    for idx, Kv in enumerate(Klist, start=1):
        minimize_from_file(directory=directory,load_file=initial,save_file=state_name + '_{:.5f}_{:.5f}.npz'.format(Kv[0], Kv[1]),
                           J=J, D=D, Kbulk=np.power(D, 2) * Kv[0], Ksurf=np.power(D, 2) * Kv[1], precision=precision,maxiter = maxiter, boundary=boundary)

def make_map_from_map(directory,save_directory,ref_file,state_name='magnpzfile',J = 1.0,D=np.tan(np.pi/10), boundary=['P','P','F'], precision = 1e-7, maxiter = None):
    K,state_name_=mfm.content(directory=directory)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    for Kv in K:
        print(f'{Kv = }')
        minimize_from_file(directory=save_directory,load_file=ref_file,save_file=state_name + '_{:.5f}_{:.5f}.npz'.format(Kv[0], Kv[1]),
                           J=J, D=D, Kbulk=np.power(D, 2) * Kv[0], Ksurf=np.power(D, 2) * Kv[1],  precision=precision,maxiter = maxiter, boundary=boundary)
