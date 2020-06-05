import numpy as np
from termcolor import colored
import sys
import os
import os.path
import time
from scipy.optimize import basinhopping

import map_color

import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import pandas as pd
import matplotlib
import map_file_manager as mfm
import map_color as mc
import scipy
from scipy.interpolate import CubicSpline
import map_file_manager as mfm



font = {'family' : 'sans-serif',
        'size'   : 12}
matplotlib.rc('font', **font)
cmap='viridis'

def double_array(x):
    y=np.array(sorted(x.tolist()))
    return np.array(sorted(y.tolist() + ((y[:-1] + y[1:]) / 2).tolist()))

def quatritize(K,fill=np.nan):
    x = np.array(sorted(list(set(K[:, 0].tolist()))))
    y = np.array(sorted(list(set(K[:, 1].tolist()))))
    m=mfm.missing(K,[x,y])
    if isinstance(fill, float) or fill == np.nan:
        m=np.concatenate([m,fill*np.ones([m.shape[0],1])],axis=1)
    else:
        m=np.concatenate([m,np.array([fill(i[0],i[1]) for i in list(m)]).reshape(-1,1)],axis=1)
    K=np.concatenate([K,m],axis=0)
    K=K.tolist()
    K=sorted(K,key=lambda x: x[:2])
    return np.array(K)

def points(x,y,energy_diff,xlim=None,ylim=None):
    eqy = []
    #plt.scatter(x.reshape(-1),y.reshape(-1),c=energy_diff.reshape(-1))
    #plt.colorbar()
    #plt.show()
    test = np.linspace(0, 20, 500)
    if xlim is None:
        xlim = np.array([x.min(),x.max()]*x.shape[0]).reshape(2,-1)
        ylim = np.array([y.min(),y.max()]*y.shape[1]).reshape(2,-1)
    print(xlim.shape)
    print(x.shape[0])
    print(ylim.shape)
    print(y.shape[1])
    for idx, row in enumerate(energy_diff.T):
        #plt.plot(y[:,idx], row)
        #plt.show()
        c = CubicSpline(y[:, idx], row)
        roots = c.roots()
        for root in roots:
            if root > ylim[0][idx] and root <= ylim[1][idx]:
                #plt.plot(test, c(test))
                #plt.title('{}{}'.format(x[0,idx],roots))
                #plt.show()
                eqy.append([x[0,idx], root])
    eqx = []
    test = np.linspace(-3, 0, 100)
    for idx, column in enumerate(energy_diff):
        #plt.plot(x[idx,:], column)
        #plt.show()
        c = CubicSpline(x[idx,:], column)
        roots = c.roots()
        for root in roots:
            if root > xlim[0][idx] and root <= xlim[1][idx]:
                eqx.append([root, y[idx,0]])
                #plt.plot(test, c(test))
                #plt.title('{}{}'.format(y[idx,0], roots))
                #plt.show()
    return np.array(eqx),np.array(eqy),energy_diff>0,energy_diff<=0

def check_state(type, energy):
    for i in np.argsort(energy):
        if type[i] != 'cone' and type[i] != None:
            return i
    for i in np.argsort(energy):
        if type[i] == 'cone':
            return i
    return np.argmin(energy)

def get_roots_2d(x,y,epu0,epu1):
    epu_d=epu1-epu0
    epu_d[np.isnan(epu_d)]=np.nanmax(np.abs(epu_d))
    epu1_nan=np.isnan(epu1)
    nnan = np.invert(epu1_nan)
    nnan = nnan.astype(float)
    nnan[epu1_nan] = np.nan
    xlim = np.array([np.nanmin(x*nnan,axis=1),np.nanmax(x*nnan,axis=1)])
    ylim = np.array([np.nanmin(y*nnan,axis=0), np.nanmax(y*nnan,axis=0)])
    return points(x,y,epu_d,xlim,ylim)


show=False

point=np.array([0,0])

directory_skt = [['/home/ivan/LC_SK/skt/cone/', '/home/ivan/LC_SK/skt/skyrmion/', '/home/ivan/LC_SK/skt/toron/']]

directory_zcone=['/home/ivan/LC_SK/skt/cone/','/home/ivan/LC_SK/spx/small_spx/cone/',
                 '/home/ivan/LC_SK/spx/small_spx_2/cone/', '/home/ivan/LC_SK/spx/alt/merge/cone/']
directory_xsp = [ '/home/ivan/LC_SK/spx/small_spx/best/', '/home/ivan/LC_SK/spx/small_spx_2/best/']
directory_tilted = [ '/home/ivan/LC_SK/spx/alt/merge/best/']
result_directory = '/home/ivan/LC_SK/result/'


localisation_criteria = 100

if not os.path.exists(result_directory):
    os.makedirs(result_directory)



data = [np.load(d + 'info/map_info_structurized.npz', allow_pickle=True) for d in directory_zcone]
K_z = np.concatenate([df['K'].reshape(-1,2) for df in data],axis=0)
epu_z = np.concatenate(np.array([df['energy_if_xsp'].reshape(-1) for df in data]))
#plt.scatter(K_z[:,0],K_z[:,1],c=epu_z)
#plt.colorbar()
#plt.show()

data = [np.load(d + 'info/map_info_structurized.npz', allow_pickle=True) for d in directory_xsp]
K_x = np.concatenate([df['K'].reshape(-1,2) for df in data],axis=0)
epu_x = np.concatenate(np.array([df['energy_if_xsp'].reshape(-1) for df in data]))
#plt.scatter(K_x[:,0],K_x[:,1],c=epu_x)
#plt.colorbar()
#plt.show()
data = [np.load(d + 'info/map_info_structurized.npz', allow_pickle=True) for d in directory_tilted]
K_t = np.concatenate([df['K'].reshape(-1,2) for df in data],axis=0)
epu_t = np.concatenate(np.array([df['energy_if_xsp'].reshape(-1) for df in data]))
#plt.scatter(K_t[:,0],K_t[:,1],c=epu_t)
#plt.colorbar()
#plt.show()
K=np.concatenate([K_z,K_x,K_t],axis=0)
x=np.array(sorted(list(set(K[:,0].tolist()))))
x=double_array(x)
x=double_array(x)
#x=double_array(x)
y=np.array(sorted(list(set(K[:,1].tolist()))))
y=double_array(y)
y=double_array(y)
#y=double_array(y)
x_grid,y_grid=np.meshgrid(x,y)

zcone_grid = scipy.interpolate.griddata(K_z,epu_z,(x_grid, y_grid), method='linear')
#plt.scatter(x_grid,y_grid,c=zcone_grid)
#plt.colorbar()
#plt.show()
xsp_grid = scipy.interpolate.griddata(K_x,epu_x,(x_grid, y_grid), method='linear')
#plt.scatter(x_grid,y_grid,c=xsp_grid)
#plt.colorbar()
#plt.show()

tilted_grid = scipy.interpolate.griddata(K_t,epu_t,(x_grid, y_grid), method='linear')
#plt.scatter(x_grid,y_grid,c=tilted_grid)
#plt.colorbar()
#plt.show()

eqx,eqy,table_p,table_n=get_roots_2d(x_grid,y_grid,zcone_grid,xsp_grid)
zx_points = np.concatenate([eqx,eqy],axis=0)
#print(zx_points)
plt.plot(eqx[:,0],eqx[:,1],'r.')
plt.plot(eqy[:,0],eqy[:,1],'r.')
#plt.plot(zx_points[:,0],zx_points[:,1],'k.')
#plt.show()
#plt.plot(x_grid[table_p].reshape(-1),y_grid[table_p].reshape(-1),'b.')
#plt.plot(x_grid[table_n].reshape(-1),y_grid[table_n].reshape(-1),'r.')
#plt.show()

eqx,eqy,table_p,table_n=get_roots_2d(x_grid,y_grid,zcone_grid,tilted_grid)
zt_points = np.concatenate([eqx,eqy],axis=0)
#print(zx_points)
plt.plot(eqx[:,0],eqx[:,1],'g.')
plt.plot(eqy[:,0],eqy[:,1],'g.')
#plt.plot(zx_points[:,0],zx_points[:,1],'k.')
#plt.show()
#plt.plot(x_grid[table_p].reshape(-1),y_grid[table_p].reshape(-1),'b.')
#plt.plot(x_grid[table_n].reshape(-1),y_grid[table_n].reshape(-1),'r.')
#plt.show()

eqx,eqy,table_p,table_n=get_roots_2d(x_grid,y_grid,tilted_grid,xsp_grid)
tx_points = np.concatenate([eqx,eqy],axis=0)
#print(zx_points)
plt.plot(eqx[:,0],eqx[:,1],'b.')
plt.plot(eqy[:,0],eqy[:,1],'b.')
#plt.plot(zx_points[:,0],zx_points[:,1],'k.')
#plt.show()
#plt.plot(x_grid[table_p].reshape(-1),y_grid[table_p].reshape(-1),'b.')
#plt.plot(x_grid[table_n].reshape(-1),y_grid[table_n].reshape(-1),'r.')
#plt.show()


for ds in directory_skt:
    data = [np.load(d + 'info/map_info_structurized.npz', allow_pickle=True) for d in ds]
    K = [df['K'] for df in data]
    assert np.all(np.array_equal(x, K[0]) for x in K)
    K = K[0]

    x = K[:,:, 0]
    y = K[:,:, 1]
    energy = np.array([df['energy'].reshape(-1) for df in data])
    state_type = np.array([df['state_type'].reshape(-1) for df in data])
    localisation = np.array([df['localisation'].reshape(-1) for df in data])
    energy[localisation > localisation_criteria] = np.nan
    energy=energy-energy[0]
    state_type[np.isnan(energy)] = np.nan

    states=np.full(energy.shape,-1)
    states[0,state_type[0]=='cone']=1
    states[1, state_type[1] == 'skyrmion'] = 1
    states[1, state_type[2] == 'skyrmion'] = 1
    states[2, state_type[1] == 'toron'] = 1
    states[2, state_type[2] == 'toron'] = 1

    cone_x,cone_y,table_p,table_n=points(x.T,y.T,states[0].reshape(K.shape[0],K.shape[1]).T)
    skyrmon_x,skyrmon_y,table_p,table_n=points(x.T,y.T,states[1].reshape(K.shape[0],K.shape[1]).T)
    toron_x,toron_y,table_p,table_n = points(x.T, y.T, states[2].reshape(K.shape[0], K.shape[1]).T)
#    plt.plot(cone_x[:, 0], cone_x[:, 1], 'k3')
#    plt.plot(cone_y[:, 0], cone_y[:, 1], 'k3')
    plt.plot(skyrmon_x[:, 0], skyrmon_x[:, 1], 'r3')
    plt.plot(skyrmon_y[:, 0], skyrmon_y[:, 1], 'r3')
    plt.plot(toron_x[:, 0], toron_x[:, 1], 'g3')
    plt.plot(toron_y[:, 0], toron_y[:, 1], 'g3')

plt.xlim([x.min(), x.max()])
plt.ylim([y.min(), y.max()])
plt.xlabel('$K_{bulk}/D^2$', fontsize=16)
plt.ylabel('$K_{surf}/D^2$', fontsize=16)
plt.savefig(result_directory + 'Best_points.pdf')
plt.savefig(result_directory + 'Best_points.eps')
if show: plt.show()
plt.close('all')
