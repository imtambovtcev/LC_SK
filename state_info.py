import numpy as np
from termcolor import colored
import sys
import os
os.environ['MAGNES_BACKEND'] = 'numpy'
import time
import magnes
import magnes.graphics
import matplotlib
from pathlib import Path
from shutil import copyfile
import map_file_manager as mfm
import pprint


font = {'family' : 'sans-serif',
        'size'   : 16}

matplotlib.rc('font', **font)

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
    return x0+x1+y0+y1

def state_info(file):
    os.environ['MAGNES_BACKEND'] = 'numpy'
    filename=Path(file)

    state_name=str.split(filename.stem, '_')[0]
    Kv=str.split(filename.stem, '_')[1:]

    container = magnes.io.load(file)
    system=container.extract_system()
    s = container["STATE"]
    size=np.array(s.shape)[:3]
    state = system.field3D()
    state.upload(s)
    try:
        energy = state.energy_contributions_sum()['total']-s.shape[0]*s.shape[1]*(s.shape[2]-2)*3-s.shape[0]*s.shape[1]*5
        energy_per_unit=energy/(s.shape[0]*s.shape[1]*s.shape[2])
    except:
        energy = np.nan
        energy_per_unit=np.nan
    try: centre_charge=toplogical_charge(system, s, int(s.shape[2] / 2))
    except: centre_charge = np.nan
    try: border_charge = toplogical_charge(system, s, 0)
    except: border_charge = np.nan
    try:            border_turn = (s[:, :, 0, :, 2] > 0.1).sum() + (s[:, :, -1, :, 2] > 0.1).sum()
    except:  border_turn =np.nan
    try:            mean_z_projection=np.sum(np.dot(s,np.array([0.,0.,1.])))/(s.size/3)
    except: mean_z_projection=np.nan
    try:            mean_z_centre_projection = np.sum(np.dot(s[:,:,int(s.shape[2]/2)], np.array([0., 0., 1.]))) / (s.shape[0]*s.shape[1])
    except: mean_z_centre_projection=np.nan
    try:            mean_z_abs_projection = np.sum(np.abs(np.dot(s, np.array([0., 0., 1.])))) / (s.size/3)
    except:mean_z_abs_projection=np.nan
    try:            local=localisation(s)
    except:local=np.nan
    #try:            xperiod = Kv[2]
    #except:()
    try:
        if local<100:
            if abs(centre_charge)<0.3 and abs(border_charge)<0.3 and border_turn<20:
                state_type='cone'
            elif abs(centre_charge-1)<0.3 and abs(border_charge)<0.3 and border_turn<20:
                state_type='toron'
            elif abs(centre_charge - 1) < 0.3 and (abs(border_charge-1) < 0.3 or border_turn > 20):
                state_type = 'skyrmion'
        else:
            state_type = np.nan
    except: state_type = np.nan
    try:
        if  True:#mean_z_abs_projection>0.5:
            energy_if_xsp = energy_per_unit
    except: energy_if_xsp = np.nan
    return {'file':file,'size':size,'J':system.exchange[0].J[0,0,0],'D':system.exchange[0].DM[0,0,0,0],'bc':system.bc,'Kv':Kv, 'K_bulk':system.anisotropy[0].strength[0,0,1,0],'K_surf':system.anisotropy[0].strength[0,0,0,0]-system.anisotropy[0].strength[0,0,1,0],
             'energy':energy,'energy_per_unit':energy_per_unit,'state_type':state_type, 'state_name':state_name,
             'localisation':local,'centre_charge':centre_charge,'border_charge':border_charge,'border_turn':border_turn,
             'mean_z_projection':mean_z_projection,'mean_z_centre_projection':mean_z_centre_projection,'mean_z_abs_projection':mean_z_abs_projection,'energy_if_xsp':energy_if_xsp}

if __name__ == "__main__":
    #pprint.pprint(state_info(sys.argv[1]))
    pprint.pprint(state_info('./skt/toron/toron_-0.90000_3.00000.npz'))