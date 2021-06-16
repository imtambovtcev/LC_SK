import numpy as np
from termcolor import colored
import sys
import os

import time
import matplotlib
from pathlib import Path
from shutil import copyfile
import map_file_manager as mfm
from scipy.interpolate import CubicSpline
import utilities
import magnes
import skyrmion_profile

from progress.bar import Bar

font = {'family' : 'sans-serif',
        'size'   : 16}

matplotlib.rc('font', **font)



def nannanargmin(energy):
    mask=np.sum(np.invert(np.isnan(energy)),axis=2)
    mask=mask>0
    minenergy=np.full(energy.shape[:2],np.nan)
    minenergy[mask]=np.nanargmin(energy[mask],axis=1)
    return minenergy,mask

def get_best(directory,compute_energies=True,compute_charge= True,compute_turns=True,compute_projections=True,
             compute_periods=True,compute_negative=True,compute_skyrmion_size='fast', compute_localisation=True):
    directory=Path(directory)
    file = np.load(str(directory.joinpath('info/map_info_structurized.npz')), allow_pickle=True)
    K=file['K']
    energy=file['energy_per_unit']
    print(f'{energy.shape = }')
    best,mask=nannanargmin(energy)
    if not os.path.exists(str(directory.joinpath('best/'))):
        os.makedirs(str(directory.joinpath('best/')))

    xperiod = np.full([K.shape[0],K.shape[1]], np.nan)
    for ct0 in range(K.shape[0]):
        for ct1 in range(K.shape[1]):
            if not np.isnan(best[ct0,ct1]):
                old_filename=Path('{}_{:.5f}_{:.5f}_{:.5f}.npz'.format(str(file['state_name']),*K[ct0,ct1,int(best[ct0,ct1])]))
                new_filename = Path('{}_{:.5f}_{:.5f}.npz'.format(str(file['state_name']), *(K[ct0, ct1, int(best[ct0, ct1])][:2])))
                copyfile(str(directory.joinpath(old_filename)),str(directory.joinpath('best/').joinpath(new_filename)))
                xperiod[ct0,ct1]=K[ct0,ct1,int(best[ct0,ct1])][2]

    map_info(str(directory.joinpath('best/')),{'xperiod':xperiod},compute_energies=compute_energies,compute_charge= compute_charge,
                                      compute_turns=compute_turns,compute_projections=compute_projections,compute_periods=compute_periods,
                                      compute_negative=compute_negative,compute_skyrmion_size=compute_skyrmion_size, compute_localisation=compute_localisation)


def epu_from(directory):
    directory=Path(directory)

    data=np.load(str(directory.joinpath('info/map_info_structurized.npz')))
    epu=data['energy_per_unit']
    try:
       if Path(directory).parent.joinpath('cone').is_dir():
           data_cone = np.load(
               str(Path(directory).parent.joinpath('cone').joinpath('info/map_info_structurized.npz')),
               allow_pickle=True)
       elif Path(directory).parent.parent.joinpath('cone').is_dir():
           data_cone = np.load(
               str(Path(directory).parent.parent.joinpath('cone').joinpath('info/map_info_structurized.npz')),
               allow_pickle=True)
       epu_cone = data_cone['energy_per_unit']
    except:
       epu_cone=np.nan
    try:
        if Path(directory).parent.joinpath('ferr').is_dir():
            data_ferr = np.load(
                str(Path(directory).parent.joinpath('ferr').joinpath('info/map_info_structurized.npz')),
                allow_pickle=True)
        elif Path(directory).parent.parent.joinpath('ferr').is_dir():
            data_ferr = np.load(
                str(Path(directory).parent.parent.joinpath('ferr').joinpath('info/map_info_structurized.npz')),
                allow_pickle=True)   
        epu_ferr = data_ferr['energy_per_unit']
    except:
       epu_ferr=np.nan
    try:
        if Path(directory).parent.joinpath('helicoid_1').is_dir():
            data_helicoid = np.load(
                str(Path(directory).parent.joinpath('helicoid_1').joinpath('best/info/map_info_structurized.npz')),
                allow_pickle=True)
        elif Path(directory).parent.parent.joinpath('helicoid_1').is_dir():
            data_helicoid = np.load(
                str(Path(directory).parent.parent.joinpath('helicoid_1').joinpath('best/info/map_info_structurized.npz')),
                allow_pickle=True)
        epu_helicoid = data_helicoid['energy_per_unit']
    except:
       epu_helicoid=np.nan
    try:
        epu_from_cone= epu-epu_cone
    except:
        epu_from_cone=np.nan
    try:
        epu_from_ferr= epu-epu_ferr
    except:
        epu_from_ferr=np.nan
    try:
        epu_from_helicoid= epu-epu_helicoid
    except:
        epu_from_helicoid=np.nan
    return (epu_from_cone,epu_from_ferr,epu_from_helicoid)



def structurize(directory,var_add,compute_energies=True,compute_charge= True,compute_turns=True,compute_projections=True,
             compute_periods=True,compute_negative=True,compute_skyrmion_size='fast', compute_localisation=True):
    directory=Path(directory)
    file = np.load(str(directory.joinpath('info/map_info.npz')),allow_pickle=True)
    K=file['K']
    Klist=[sorted(list(dict.fromkeys(i.tolist()))) for i in K.T]
    Kshape = [len(i) for i in Klist]
    full_K=np.full(Kshape+[len(Klist)],np.nan)
    var = {}
    var0 = {}
    for f in file.files:
        if f != 'K' and f != 'allow_pickle':
            if file[f].shape!=():
                if f == 'state_type':
                    var[f] = np.full(Kshape, np.nan, dtype=object)
                else:
                    var[f] = np.full(Kshape,np.nan)

            else:
                var0[f]=file[f]
    K=K.tolist()
    if len(Klist)==2:
        for idx0,K0 in enumerate(Klist[0]):
            for idx1, K1 in enumerate(Klist[1]):
                full_K[idx0,idx1,:]=[K0, K1]
                if [K0, K1] in K:
                    idx = K.index([K0, K1])
                    for i in var:
                        #print(i)
                        #print(f'{var[i] = }\t {file[i] = }')
                        var[i][idx0, idx1] = file[i][idx]

    if len(Klist)==3:
        for idx0,K0 in enumerate(Klist[0]):
            for idx1, K1 in enumerate(Klist[1]):
                for idx2, K2 in enumerate(Klist[2]):
                    full_K[idx0, idx1,idx2, :] = [K0, K1, K2]
                    if [K0,K1,K2] in K:
                        idx=K.index([K0,K1,K2])
                        #print(f'{idx = }\t{idx0 =}\t{idx1 = }\t{idx2 = }')
                        for i in var:
                            var[i][idx0,idx1,idx2] = file[i][idx]

    print(f'{full_K.shape = }')
    np.savez(str(directory.joinpath('info/map_info_structurized.npz')),K=full_K,**{**var0,**var,**var_add},allow_pickle=True)
    if len(Klist)==3:
        get_best(str(directory),compute_energies=compute_energies,compute_charge= compute_charge,
                                      compute_turns=compute_turns,compute_projections=compute_projections,compute_periods=compute_periods,
                                      compute_negative=compute_negative,compute_skyrmion_size=compute_skyrmion_size, compute_localisation=compute_localisation)
    if len(Klist)==2:
        epu_from_cone,epu_from_ferr,epu_from_helicoid=epu_from(str(directory))
        var_add2={'epu_from_cone':epu_from_cone,'epu_from_ferr':epu_from_ferr,'epu_from_helicoid':epu_from_helicoid}
        np.savez(str(directory.joinpath('info/map_info_structurized.npz')),K=full_K,**{**var0,**var,**var_add,**var_add2},allow_pickle=True)


def toplogical_charge(system,s,layer_n):
    layer = s[:, :, layer_n, :, :].reshape([system.size[0], system.size[1], 3])
    dx = np.diff(np.concatenate((layer, layer[-1, :, :].reshape(1, system.size[1], 3)), axis=0), axis=0)
    dy = np.diff(np.concatenate((layer, layer[:, -1, :].reshape(system.size[1], 1, 3)), axis=1), axis=1)
    return 3*np.sum(np.cross(dx,dy)*layer)/(4*np.pi*np.pi)

def localisation(state):
    x0=state[0,:,:,:,:]
    x0=x0-x0[0]
    x0=np.linalg.norm(x0)
    x1 = state[-1, :, :, :, :]
    x1 = x1 - x1[0]
    x1 = np.linalg.norm(x1)
    y0 = state[:, 0, :, :, :]
    y0 = y0 - y0[0]
    y0 = np.linalg.norm(y0)
    y1 = state[:, -1, :, :, :]
    y1 = y1 - y1[0]
    y1 = np.linalg.norm(y1)
    return (x0+x1+y0+y1)/(2*(state.shape[0]+state.shape[1])*state.shape[2])

def angle(s):
    return 0

def state_info(filename,compute_energies=True,compute_charge= False,compute_turns=False,compute_projections=False,
             compute_periods=True,compute_negative=False,compute_skyrmion_size=False, compute_localisation=False):
    data={}
    container = magnes.io.load(str(filename))
    system = container.extract_system()
    if "PATH" in container:
        s = list(container["PATH"])[0]
        # print('state from path')
    else:
        s = container["STATE"]
        # print('state from state')
    state = system.field3D()
    state.upload(s)
    if compute_energies:
        try:
            data['energy'] = state.energy_contributions_sum()['total'] \
                          - s.shape[0] * s.shape[1] * (s.shape[2] - 2) * 3 - s.shape[0] * s.shape[1] * 5
            data['energy_per_unit'] = data['energy'] / (s.shape[0] * s.shape[1] * s.shape[2])
        except:
            ()
        try:
            data['energy_original'] = state.energy_contributions_sum()['total']
        except:
            ()
        try:
            data['energy_J'] = state.energy_contributions_sum()['Heisenberg']
        except:
            ()
        try:
            data['energy_D'] = state.energy_contributions_sum()['DM']
        except:
            ()
        try:
            data['energy_K'] = state.energy_contributions_sum()['anisotropy']
        except:
            ()
        try:
            data['energy_H'] = state.energy_contributions_sum()['Zeeman']
        except:
            ()
        try:
            if True:
                data['energy_if_xsp'] = data['energy_per_unit']
        except:
            ()
    if compute_charge:
        try:
            data['centre_charge'] = toplogical_charge(system, s, int(s.shape[2] / 2))
        except:
            ()
        try:
            data['border_charge'] = toplogical_charge(system, s, 0)
        except:
            ()
    if compute_turns:
        try:
            data['border_turn'] = (s[:, :, 0, :, 2] > 0.1).sum() + (s[:, :, -1, :, 2] > 0.1).sum()
        except:
            ()
    if compute_projections:
        try:
            data['mean_z_projection'] = np.sum(np.dot(s, np.array([0., 0., 1.]))) / (s.size / 3)
        except:
            ()
        try:
            data['mean_z_centre_projection'] = np.sum(np.dot(s[:, :, int(s.shape[2] / 2)],
                                                          np.array([0., 0., 1.]))) / (s.shape[0] * s.shape[1])
        except:
            ()
        try:
            data['mean_z_centre_abs_projection'] = np.sum(np.abs(np.dot(s[:, :, int(s.shape[2] / 2)],
                                                                     np.array([0., 0., 1.])))) / (
                                                            s.shape[0] * s.shape[1])
        except:
            ()
        try:
            data['mean_x_projection'] = np.sum(np.dot(s, np.array([1., 0., 0.]))) / (s.size / 3)
        except:
            ()
        try:
            data['mean_x_abs_projection'] = np.sum(np.abs(np.dot(s, np.array([1., 0., 0.])))) / (s.size / 3)
        except:
            ()
        try:
            data['mean_x_centre_abs_projection'] = np.sum(np.abs(np.dot(s[:, :, int(s.shape[2] / 2)],
                                                                     np.array([1., 0., 0.])))) / (
                                                            s.shape[0] * s.shape[1])
        except:
            ()
        try:
            data['mean_x_centre_abs_projection_angle'] = np.arccos(data['mean_x_centre_abs_projection']) * 360 / (2 * np.pi)
        except:
            ()
        try:
            data['mean_z_abs_projection'] = np.sum(np.abs(np.dot(s, np.array([0., 0., 1.])))) / (s.size / 3)
        except:
            ()
    if compute_periods:
        try:
            data['angle'] = utilities.get_angle(s)
        except:
            ()
        try:
            data['zperiod'] = utilities.get_zperiod(s)
        except:
            ()
        try:
            data['zturns'] = np.min([s.shape[2] / data['zperiod'], 5])
        except:
            ()
        try:
            data['xperiod'] = s.shape[0]
        except:
            ()
        try:
            data['x_tilted_period'] = data['xperiod'] * np.sin(data['angle'] * np.pi / 180)
        except:
            ()
    if compute_skyrmion_size:
        try:
            if compute_skyrmion_size == 'fast':
                # print('fast skyrmion size')
                data['skyrmion_size'], data['bober_size'], data['topbober_size'], data['bottombober_size'] = skyrmion_profile.fast_skyrmion_size_compute(s)
                if data['skyrmion_size'] > 0.1 and data['bober_size'] > 1.:
                    data['is_skyrmion'] = True
                elif data['skyrmion_size'] > 0.1 and data['bober_size'] <= 1.:
                    data['is_toron'] = True
                elif data['skyrmion_size'] <= 0.1 and data['bober_size'] > 1.:
                    data['is_bober'] = True
                if data['skyrmion_size'] > 0.1 and np.logical_xor(data['topbober_size'] > 1.,data['bottombober_size'] > 1.) :
                    data['is_leech'] = True
                # print(f'{skyrmion_size'] = }\t{bober_size'] = }')
            elif compute_skyrmion_size == 'full':
                # print('full skyrmion size')
                r_min, r_max, skyrmion_mask_sum = skyrmion_profile.skyrmion_profile(filename, show=False)
                data['skyrmion_size'] = r_max[int(s.shape[2] / 2)]
                if np.isnan(data['skyrmion_size']):
                    data['skyrmion_size'] = 0.
                # print(f'{r_max = }')
                # print(f'{data['skyrmion_size'] = }')
        except:
            ()
    if compute_negative:
        try:
            print('computing negative')
            rate, negative, normal2d, oper, first, second = \
                magnes.rate.dynamic_prefactor_htst(system, state, normal=None, tolerance=1e-5, number_of_modes=1)
            negative = np.array(negative)[0]
            data['smallest_eigenvalue'] = negative
            data['eigenvalue_positive'] = 1 if negative >= 0 else 0
            print(f'{negative = }')
        except:
            ()

    if compute_localisation:
        try:
            data['local'] = localisation(s)
        except:
            ()
        try:
            if data['local'] < 1:
                if abs(data['centre_charge']) < 0.3 and abs(data['border_charge']) < 0.3 and data['border_turn'] < 20:
                    data['state_type'] = 'cone'
                elif abs(data['centre_charge']) < 0.3 and (
                        abs(abs(data['border_charge']) - 1) < 0.3 or data['border_turn'] > 20):
                    data['state_type'] = 'bober'
                elif abs(abs(data['centre_charge']) - 1) < 0.3 and abs(data['border_charge']) < 0.3 and data['border_turn'] < 20:
                    data['state_type'] = 'toron'
                elif abs(abs(data['centre_charge']) - 1) < 0.3 and (
                        abs(abs(data['border_charge']) - 1) < 0.3 or data['border_turn'] > 20):
                    data['state_type'] = 'skyrmion'
            else:
                print(f'{data["local"] = }')
                data['state_type'] = np.nan
        except:
            ()
    return data

def map_info(directory,var={},compute_energies=True,compute_charge= True,compute_turns=True,compute_projections=True,
             compute_periods=True,compute_negative=True,compute_skyrmion_size='fast', compute_localisation=True):
    print(f'{var = },{bool(var) = }')
    directory=Path(directory)
    K, state_name=mfm.content(str(directory))

    state_type=np.full(K.shape[0],np.nan,dtype=object)
    energy=np.full(K.shape[0],np.nan)
    energy_original = np.full(K.shape[0], np.nan)
    energy_J = np.full(K.shape[0], np.nan)
    energy_D = np.full(K.shape[0], np.nan)
    energy_K = np.full(K.shape[0], np.nan)
    energy_H = np.full(K.shape[0], np.nan)
    energy_per_unit = np.full(K.shape[0], np.nan)
    epu_from_cone = np.full(K.shape[0], np.nan)
    epu_from_ferr = np.full(K.shape[0], np.nan)
    epu_from_helicoid = np.full(K.shape[0], np.nan)
    local=np.full(K.shape[0],np.nan)
    centre_charge=np.full(K.shape[0],np.nan)
    border_charge=np.full(K.shape[0],np.nan)
    border_turn=np.full(K.shape[0],np.nan)
    mean_z_projection=np.full(K.shape[0],np.nan)
    mean_z_abs_projection = np.full(K.shape[0], np.nan)
    mean_z_centre_projection = np.full(K.shape[0], np.nan)
    mean_z_centre_abs_projection = np.full(K.shape[0], np.nan)
    mean_x_projection = np.full(K.shape[0], np.nan)
    mean_x_abs_projection = np.full(K.shape[0], np.nan)
    mean_x_centre_abs_projection = np.full(K.shape[0], np.nan)
    mean_x_centre_abs_projection_angle = np.full(K.shape[0], np.nan)
    angle = np.full(K.shape[0], np.nan)
    zperiod = np.full(K.shape[0], np.nan)
    zturns = np.full(K.shape[0], np.nan)
    energy_if_xsp = np.full(K.shape[0], np.nan)
    xperiod = np.full(K.shape[0], np.nan)
    x_tilted_period = np.full(K.shape[0], np.nan)

    smallest_eigenvalue=np.full(K.shape[0], np.nan)
    eigenvalue_positive=np.full(K.shape[0], np.nan)
    skyrmion_size = np.full(K.shape[0], np.nan)
    topbober_size = np.full(K.shape[0], np.nan)
    bottombober_size = np.full(K.shape[0], np.nan)
    bober_size = np.full(K.shape[0], np.nan)
    is_skyrmion = np.full(K.shape[0], False)
    is_toron = np.full(K.shape[0], False)
    is_bober = np.full(K.shape[0], False)
    is_leech = np.full(K.shape[0], False)
    t0=time.time()
    bar_max=len(K)
    with Bar('Processing', max=bar_max, suffix='%(index)d / %(max)d [%(elapsed_td)s / %(eta_td)s]') as bar:
        for idx,Kv in enumerate(K):
            filename=state_name
            for f in Kv:
                filename+='_{:.5f}'.format(f)
            filename=Path(filename+'.npz')
            try:
                state_data=state_info(os.path.join(directory, filename),compute_energies=compute_energies,compute_charge= compute_charge,
                                      compute_turns=compute_turns,compute_projections=compute_projections,compute_periods=compute_periods,
                                      compute_negative=compute_negative,compute_skyrmion_size=compute_skyrmion_size, compute_localisation=compute_localisation)
                try:
                    state_type[idx] = state_data['state_type']
                except:()
                try:
                    energy[idx] = state_data['energy']
                except:()
                try:
                    energy_original[idx] = state_data['energy_original']
                except:
                    ()
                try:
                    energy_J[idx] = state_data['energy_J']
                except:
                    ()
                try:
                    energy_D[idx] = state_data['energy_D']
                except:
                    ()
                try:
                    energy_K[idx] = state_data['energy_K']
                except:
                    ()
                try:
                    energy_H[idx] = state_data['energy_H']
                except:
                    ()
                try:
                    energy_per_unit[idx] = state_data['energy_per_unit']
                except:
                    ()
                try:
                    epu_from_cone[idx] = state_data['epu_from_cone']
                except:
                    ()
                try:
                    epu_from_ferr[idx] = state_data['epu_from_ferr']
                except:
                    ()
                try:
                    epu_from_helicoid[idx] = state_data['epu_from_helicoid']
                except:
                    ()
                try:
                    local[idx] = state_data['local']
                except:
                    ()
                try:
                    centre_charge[idx] = state_data['centre_charge']
                except:
                    ()
                try:
                    border_charge[idx] = state_data['border_charge']
                except:
                    ()
                try:
                    border_turn[idx] = state_data['border_turn']
                except:
                    ()
                try:
                    mean_z_projection[idx] = state_data['mean_z_projection']
                except:
                    ()
                try:
                    mean_z_abs_projection[idx] = state_data['mean_z_abs_projection']
                except:
                    ()
                try:
                    mean_z_centre_projection[idx] = state_data['mean_z_centre_projection']
                except:
                    ()
                try:
                    mean_z_centre_abs_projection[idx] = state_data['mean_z_centre_abs_projection']
                except:
                    ()
                try:
                    mean_x_projection[idx] = state_data['mean_x_projection']
                except:
                    ()
                try:
                    mean_x_abs_projection[idx] = state_data['mean_x_abs_projection']
                except:
                    ()
                try:
                    mean_x_centre_abs_projection[idx] = state_data['mean_x_centre_abs_projection']
                except:
                    ()
                try:
                    mean_x_centre_abs_projection_angle[idx] = state_data['mean_x_centre_abs_projection_angle']
                except:
                    ()
                try:
                    angle[idx] = state_data['angle']
                except:
                    ()
                try:
                    zperiod[idx] = state_data['zperiod']
                except:
                    ()
                try:
                    zturns[idx] = state_data['zturns']
                except:
                    ()
                try:
                    energy_if_xsp[idx] = state_data['energy_if_xsp']
                except:
                    ()
                try:
                    xperiod[idx] = state_data['xperiod']
                except:
                    ()
                try:
                    x_tilted_period[idx] = state_data['x_tilted_period']
                except:
                    ()
                try:
                    smallest_eigenvalue[idx] = state_data['smallest_eigenvalue']
                except:
                    ()
                try:
                    eigenvalue_positive[idx] = state_data['eigenvalue_positive']
                except:
                    ()
                try:
                    skyrmion_size[idx] = state_data['skyrmion_size']
                except:
                    ()
                try:
                    topbober_size[idx] = state_data['topbober_size']
                except:
                    ()
                try:
                    bottombober_size[idx] = state_data['bottombober_size']
                except:
                    ()
                try:
                    bober_size[idx] = state_data['bober_size']
                except:
                    ()
                try:
                    is_skyrmion[idx] = state_data['is_skyrmion']
                except:
                    ()
                try:
                    is_toron[idx] = state_data['is_toron']
                except:
                    ()
                try:
                    is_bober[idx] = state_data['is_bober']
                except:
                    ()
                try:
                    is_leech[idx] = state_data['is_leech']
                except:
                    ()

            except:
                energy[idx] = np.nan
                state_type[idx] = np.nan
                print(colored('no/damaged file','red'))
            bar.next()

    if not os.path.exists(str(directory.joinpath('info/'))):
        os.makedirs(str(directory.joinpath('info/')))
    print(f'{K.shape = },{energy_if_xsp.shape = }')
    np.savez(str(directory.joinpath('info/map_info.npz')), K=K,
             energy=energy,energy_per_unit=energy_per_unit,energy_original=energy_original,
             energy_J=energy_J,energy_D=energy_D,energy_K=energy_K,energy_H=energy_H,
             state_type=state_type, state_name=state_name,
             localisation=local,centre_charge=centre_charge,border_charge=border_charge,border_turn=border_turn,
             mean_z_projection=mean_z_projection,mean_z_centre_projection=mean_z_centre_projection,
             mean_z_abs_projection=mean_z_abs_projection,mean_z_centre_abs_projection=mean_z_centre_abs_projection,
             mean_x_projection=mean_x_projection,mean_x_abs_projection=mean_x_abs_projection,
             mean_x_centre_abs_projection=mean_x_centre_abs_projection,
             mean_x_centre_abs_projection_angle=mean_x_centre_abs_projection_angle,angle=angle,zperiod=zperiod,zturns=zturns,
             energy_if_xsp=energy_if_xsp,xperiod=xperiod,x_tilted_period =x_tilted_period,
             epu_from_cone=epu_from_cone,epu_from_ferr=epu_from_ferr,epu_from_helicoid=epu_from_helicoid,
             smallest_eigenvalue=smallest_eigenvalue,eigenvalue_positive=eigenvalue_positive,
             skyrmion_size=skyrmion_size,bober_size=bober_size, topbober_size=topbober_size,bottombober_size=bottombober_size,
             is_skyrmion = is_skyrmion, is_toron=is_toron,is_bober=is_bober, is_leech=is_leech,
             allow_pickle=True)
    structurize(str(directory),var,compute_energies=compute_energies,compute_charge= compute_charge,
                                      compute_turns=compute_turns,compute_projections=compute_projections,compute_periods=compute_periods,
                                      compute_negative=compute_negative,compute_skyrmion_size=compute_skyrmion_size, compute_localisation=compute_localisation)

if __name__ == "__main__":
    directory = Path('./') if len(sys.argv) <= 1 else Path(sys.argv[1])
    compute_negative= False if len(sys.argv) <= 2 else sys.argv[2]=='True'
    compute_skyrmion_size = False if len(sys.argv) <= 3 else sys.argv[3]
    map_info(directory,compute_negative=compute_negative,compute_skyrmion_size=compute_skyrmion_size)