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
from scipy.interpolate import CubicSpline



font = {'family' : 'sans-serif',
        'size'   : 12}
matplotlib.rc('font', **font)
cmap='viridis'

def points(x,y,energy_diff):
    eqy = []
    for idx, row in enumerate(energy_diff):
        roots = CubicSpline(y[idx, :], row).roots()
        for root in roots:
            if root > y[idx, :].min() and root <= y[idx, :].max():
                eqy.append([x[idx, 0], root])
    eqx = []
    for idx, column in enumerate(energy_diff.T):
        roots = CubicSpline(x[:, idx], column).roots()
        for root in roots:
            if root > x[:, idx].min() and root <= x[:, idx].max():
                eqx.append([root, y[0, idx]])
    return np.array(eqx),np.array(eqy)

def check_state(type, energy):
    for i in np.argsort(energy):
        if type[i] != 'cone' and type[i] != None:
            return i
    for i in np.argsort(energy):
        if type[i] == 'cone':
            return i
    return np.argmin(energy)

show=False

point=np.array([0,0])

directory_skt = [['/home/ivan/LC_SK/skt/cone/', '/home/ivan/LC_SK/skt/skyrmion/', '/home/ivan/LC_SK/skt/toron/']]

directory_best = [['/home/ivan/LC_SK/spx/alt/merge/cone/','/home/ivan/LC_SK/spx/alt/merge/best/']]
result_directory = '/home/ivan/LC_SK/result/'

xlimits=[-3,0]
ylimits=[0,30]

localisation_criteria = 100

if not os.path.exists(result_directory):
    os.makedirs(result_directory)

#for d in directory:
#    map_color.map_color(d,show=show)
best_points=[]
for db in directory_best:
    data = [np.load(d + 'info/map_info_structurized.npz', allow_pickle=True) for d in db]
    K = [df['K'] for df in data]
    assert np.all(np.array_equal(x, K[0]) for x in K)
    K = K[0]

    epu = [df['energy_if_xsp'] for df in data]
    x = K[:,:, 0]
    y = K[:,:, 1]
    energy_diff=epu[1]-epu[0]
    energy_diff[np.isnan(energy_diff)] = 2 * np.nanmax(energy_diff)
    eqx,eqy=points(x,y,energy_diff)
    best_points+=eqx.tolist()
    best_points+=eqy.tolist()
best_points=sorted(best_points)
best_points=np.array(best_points)
#best_points=best_points[best_points[:,1]>5]
plt.plot(best_points[:,0],best_points[:,1],'b.')
#plt.plot(best_points[:,0],best_points[:,1],'b--')

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

    cone_x,cone_y=points(x,y,states[0].reshape(K.shape[0],K.shape[1]))
    skyrmon_x,skyrmon_y=points(x,y,states[1].reshape(K.shape[0],K.shape[1]))
    toron_x,toron_y = points(x, y, states[2].reshape(K.shape[0], K.shape[1]))
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
if xlimits:
    plt.xlim(xlimits)
if ylimits:
    plt.ylim(ylimits)
plt.savefig(result_directory + 'Best_points.pdf')
plt.savefig(result_directory + 'Best_points.eps')
if show: plt.show()
plt.close('all')
