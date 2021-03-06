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

def get_best(directory):
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

    map_info(str(directory.joinpath('best/')),{'xperiod':xperiod})


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
        epu_from_cone= epu-epu_cone
    except:
        epu_from_cone=np.nan
    try:
        epu_from_ferr= epu-epu_ferr
    except:
        epu_from_ferr=np.nan
    return (epu_from_cone,epu_from_ferr)



def structurize(directory,var_add):
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
        get_best(str(directory))
    if len(Klist)==2:
        epu_from_cone,epu_from_ferr=epu_from(str(directory))
        var_add2={'epu_from_cone':epu_from_cone,'epu_from_ferr':epu_from_ferr}
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

def map_info(directory,var={},compute_energies=False,compute_charge= False,compute_turns=False,compute_projections=False,
             compute_periods= False,compute_negative=False,compute_skyrmion_size=False, compute_localisation=False):
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
    bober_size = np.full(K.shape[0], np.nan)
    is_skyrmion = np.full(K.shape[0], False)
    is_toron = np.full(K.shape[0], False)
    is_bober = np.full(K.shape[0], False)
    t0=time.time()
    bar_max=len(K)
    with Bar('Processing', max=bar_max, suffix='%(index)d / %(max)d [%(elapsed_td)s / %(eta_td)s]') as bar:
        for idx,Kv in enumerate(K):
            filename=state_name
            for f in Kv:
                filename+='_{:.5f}'.format(f)
            filename=Path(filename+'.npz')
            try:
                container = magnes.io.load(str(os.path.join(directory, filename)))
                system=container.extract_system()
                if "PATH" in container:
                    s = list(container["PATH"])[0]
                    #print('state from path')
                else:
                    s = container["STATE"]
                    #print('state from state')
                state = system.field3D()
                state.upload(s)
                if compute_energies:
                    try:
                        energy[idx] = state.energy_contributions_sum()['total']\
                                      -s.shape[0]*s.shape[1]*(s.shape[2]-2)*3-s.shape[0]*s.shape[1]*5
                        energy_per_unit[idx]=energy[idx]/(s.shape[0]*s.shape[1]*s.shape[2])
                    except:()
                    try:
                        energy_original[idx] = state.energy_contributions_sum()['total']
                    except:()
                    try:
                        energy_J[idx] = state.energy_contributions_sum()['Heisenberg']
                    except:()
                    try:
                        energy_D[idx] = state.energy_contributions_sum()['DM']
                    except:()
                    try:
                        energy_K[idx] = state.energy_contributions_sum()['anisotropy']
                    except:()
                    try:
                        energy_H[idx] = state.energy_contributions_sum()['Zeeman']
                    except:()
                    try:
                        if True:  # mean_z_abs_projection[idx]>0.5:
                            energy_if_xsp[idx] = energy_per_unit[idx]
                    except:
                        ()
                if compute_charge:
                    try: centre_charge[idx]=toplogical_charge(system, s, int(s.shape[2] / 2))
                    except:()
                    try: border_charge[idx] = toplogical_charge(system, s, 0)
                    except:()
                if compute_turns:
                    try:            border_turn[idx] = (s[:, :, 0, :, 2] > 0.1).sum() + (s[:, :, -1, :, 2] > 0.1).sum()
                    except:()
                if compute_projections:
                    try:            mean_z_projection[idx]=np.sum(np.dot(s,np.array([0.,0.,1.])))/(s.size/3)
                    except:()
                    try:            mean_z_centre_projection[idx] = np.sum(np.dot(s[:,:,int(s.shape[2]/2)],
                                                                                  np.array([0., 0., 1.]))) / (s.shape[0]*s.shape[1])
                    except:()
                    try:            mean_z_centre_abs_projection[idx] = np.sum(np.abs(np.dot(s[:,:,int(s.shape[2]/2)],
                                                                                  np.array([0., 0., 1.])))) / (s.shape[0]*s.shape[1])
                    except:()
                    try:            mean_x_projection[idx]=np.sum(np.dot(s,np.array([1.,0.,0.])))/(s.size/3)
                    except:()
                    try:            mean_x_abs_projection[idx] = np.sum(np.abs(np.dot(s,np.array([1.,0.,0.]))))/(s.size/3)
                    except:()
                    try:            mean_x_centre_abs_projection[idx] = np.sum(np.abs(np.dot(s[:,:,int(s.shape[2]/2)],
                                                                                  np.array([1., 0., 0.])))) / (s.shape[0]*s.shape[1])
                    except:()
                    try:            mean_x_centre_abs_projection_angle[idx] = np.arccos(mean_x_centre_abs_projection[idx])*360/(2*np.pi)
                    except:()
                    try:
                        mean_z_abs_projection[idx] = np.sum(np.abs(np.dot(s, np.array([0., 0., 1.])))) / (s.size / 3)
                    except:
                        ()
                if compute_periods:
                    try:            angle[idx] = utilities.get_angle(s)
                    except:()
                    try:            zperiod[idx] = utilities.get_zperiod(s)
                    except:()
                    try:            zturns[idx] = np.min([s.shape[2]/zperiod[idx],5])
                    except:()
                    try:
                        xperiod[idx] = s.shape[0]
                    except:
                        ()
                    try:
                        x_tilted_period[idx] = xperiod[idx] * np.sin(angle[idx] * np.pi / 180)
                    except:
                        ()
                if compute_skyrmion_size:
                    try:
                        if compute_skyrmion_size == 'fast':
                            #print('fast skyrmion size')
                            skyrmion_size[idx],bober_size[idx] = skyrmion_profile.fast_skyrmion_size_compute(s)
                            if skyrmion_size[idx]>0.1 and bober_size[idx]>1.:
                                is_skyrmion[idx]=True
                                print(f'{is_skyrmion[idx] = }')
                            elif skyrmion_size[idx]>0.1 and bober_size[idx]<=1.:
                                is_toron[idx] = True
                                print(f'{is_toron[idx] = }')
                            elif skyrmion_size[idx]<=0.1 and bober_size[idx]>1.:
                                is_bober[idx] = True
                                print(f'{is_bober[idx] = }')
                            #print(f'{skyrmion_size[idx] = }\t{bober_size[idx] = }')
                        elif compute_skyrmion_size == 'full':
                            #print('full skyrmion size')
                            r_min,r_max,skyrmion_mask_sum=skyrmion_profile.skyrmion_profile(filename, show=False)
                            skyrmion_size[idx]=r_max[int(s.shape[2]/2)]
                            if np.isnan(skyrmion_size[idx]):
                                skyrmion_size[idx]=0.
                            #print(f'{r_max = }')
                            #print(f'{skyrmion_size[idx] = }')
                    except:()
                if compute_negative:
                    try:
                        print('computing negative')
                        rate, negative, normal2d, oper, first, second = \
                            magnes.rate.dynamic_prefactor_htst(system, state, normal=None,tolerance=1e-5, number_of_modes=1)
                        negative=np.array(negative)[0]
                        smallest_eigenvalue[idx]=negative
                        eigenvalue_positive[idx]=1 if negative>=0 else 0
                        print(f'{negative = }')
                    except:                ()

                if compute_localisation:
                    try:  local[idx]=localisation(s)
                    except:()
                    try:
                        if local[idx]<1:
                            if abs(centre_charge[idx])<0.3 and abs(border_charge[idx])<0.3 and border_turn[idx]<20:
                                state_type[idx]='cone'
                            elif abs(centre_charge[idx]) < 0.3 and (abs(abs(border_charge[idx])-1) < 0.3 or border_turn[idx] > 20):
                                state_type[idx] = 'bober'
                            elif abs(abs(centre_charge[idx])-1)<0.3 and abs(border_charge[idx])<0.3 and border_turn[idx]<20:
                                state_type[idx]='toron'
                            elif abs(abs(centre_charge[idx]) - 1) < 0.3 and (abs(abs(border_charge[idx])-1) < 0.3 or border_turn[idx] > 20):
                                state_type[idx] = 'skyrmion'
                        else:
                            print(f'{local[idx] = }')
                            state_type[idx] = np.nan
                    except: ()
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
             epu_from_cone=epu_from_cone,epu_from_ferr=epu_from_ferr,
             smallest_eigenvalue=smallest_eigenvalue,eigenvalue_positive=eigenvalue_positive,
             skyrmion_size=skyrmion_size,bober_size=bober_size, is_skyrmion = is_skyrmion, is_toron=is_toron,is_bober=is_bober,
             allow_pickle=True)
    structurize(str(directory),var)

if __name__ == "__main__":
    directory = Path('./') if len(sys.argv) <= 1 else Path(sys.argv[1])
    compute_negative= False if len(sys.argv) <= 2 else sys.argv[2]=='True'
    compute_skyrmion_size = False if len(sys.argv) <= 3 else sys.argv[3]
    map_info(directory,compute_negative=compute_negative,compute_skyrmion_size=compute_skyrmion_size)