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
from pathlib import Path



font = {'family' : 'sans-serif',
        'size'   : 12}
matplotlib.rc('font', **font)
cmap='viridis'

def point_plot(directory,x_all,y_all,z_all,point,name,show=False):
    for x,y,z,n in zip(x_all,y_all,z_all,name):
        point_column = np.argmin(np.linalg.norm(x - point[0], axis=1))
        point_row = np.argmin(np.linalg.norm(y - point[1], axis=0))
        print('Point = ', x[point_column][0], y[:, point_row][0])
        axisn = point_column
        z1 = z[axisn, :]
        px = y[axisn, np.invert(np.isnan(z1))]
        py = z1[np.invert(np.isnan(z1))]
        if len(py) > 1:
            plt.plot(px, py, '.',label=n)
    plt.xlabel('$K_{surf}/D^2$', fontsize=16)
    plt.ylabel('xperiod', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory + '_surf.pdf')
    if show: plt.show()
    plt.close('all')
    for x,y,z,n in zip(x_all,y_all,z_all,name):
        point_column = np.argmin(np.linalg.norm(x - point[0], axis=1))
        point_row = np.argmin(np.linalg.norm(y - point[1], axis=0))
        print('Point = ', x[point_column][0], y[:, point_row][0])
        axisn = point_column
        z1 = z[axisn, :]
        px = y[axisn, np.invert(np.isnan(z1))]
        py = z1[np.invert(np.isnan(z1))]
        if len(py) > 1:
            int = np.poly1d(np.polyfit(px, py, 5))
            ipx = np.linspace(px.min(), 50, num=100, endpoint=True)
            ipy = int(ipx)

            plt.plot(ipx, ipy,label=n)
    plt.xlabel('$K_{surf}/D^2$', fontsize=16)
    plt.ylabel('xperiod', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory+ '_pi_surf.pdf')
    if show: plt.show()
    plt.close('all')

    for x,y,z,n in zip(x_all,y_all,z_all,name):
        point_column = np.argmin(np.linalg.norm(x - point[0], axis=1))
        point_row = np.argmin(np.linalg.norm(y - point[1], axis=0))
        print('Point = ', x[point_column][0], y[:, point_row][0])
        axisn = point_column
        axisn = point_row
        z2 = z[:, axisn]
        px = x[np.invert(np.isnan(z2)), axisn]
        py = z2[np.invert(np.isnan(z2))]
        if len(py) > 1:
            plt.plot(px, py, '.',label=n)

    plt.xlabel('$K_{bulk}/D^2$', fontsize=16)
    plt.ylabel('xperiod', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory + '_p_bulk.pdf')
    if show: plt.show()
    plt.close('all')

    for x,y,z,n in zip(x_all,y_all,z_all,name):
        point_column = np.argmin(np.linalg.norm(x - point[0], axis=1))
        point_row = np.argmin(np.linalg.norm(y - point[1], axis=0))
        print('Point = ', x[point_column][0], y[:, point_row][0])
        axisn = point_column
        axisn = point_row
        z2 = z[:, axisn]
        px = x[np.invert(np.isnan(z2)), axisn]
        py = z2[np.invert(np.isnan(z2))]
        if len(py) > 1:
            int = np.poly1d(np.polyfit(px, py, 2))
            ipx = np.linspace(-0.3, px.max(), num=100, endpoint=True)
            ipy = int(ipx)
            plt.plot(ipx, ipy,label=n)
    plt.xlabel('$K_{bulk}/D^2$', fontsize=16)
    plt.ylabel('xperiod', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory + '_pi_bulk.pdf')
    if show: plt.show()
    plt.close('all')

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

point=np.array([0,10])

'''
directory = ['/home/ivan/LC_SK/skt/cone/', '/home/ivan/LC_SK/skt/skyrmion/', '/home/ivan/LC_SK/skt/toron/']
result_directory = '/home/ivan/LC_SK/skt/map_result/'
mode='skt'
'''
'''
directory = ['/home/ivan/LC_SK/spx/mat_cone/', '/home/ivan/LC_SK/spx/spx_map/best/']
result_directory = '/home/ivan/LC_SK/spx/mat_map/'
mode='best'
'''

directory = ['/home/ivan/LC_SK/spx/alt/10/cone/', '/home/ivan/LC_SK/spx/alt/10/best/']
result_directory = '/home/ivan/LC_SK/spx/alt/10/best/result/'
mode='best'

'''
#'/home/ivan/LC_SK/spx/spx_bulk_1/best/',
directory = [ '/home/ivan/LC_SK/spx/spx_surf_10/best/',
              '/home/ivan/LC_SK/spx/spx_surf_20/best/', '/home/ivan/LC_SK/spx/spx_surf_40/best/']
result_directory = '/home/ivan/LC_SK/spx/spx_surf_result/'
mode='compare'
point = np.array([0., 0.])
'''

localisation_criteria = 100

if not os.path.exists(result_directory):
    os.makedirs(result_directory)

#for d in directory:
#    map_color.map_color(d,show=show)



if mode=='compare':
    data = [np.load(d + 'info/map_info_structurized.npz', allow_pickle=True) for d in directory]
    K = [df['K'] for df in data]
    xperiod=[df['xperiod'] for df in data]
    x = [Ki[:, :, 0] for Ki in K ]
    y = [Ki[:, :, 1] for Ki in K]
    name=[Path(i).parts[-2].split('_')[-1] for i in directory]
    point_plot(result_directory + 'xperiod.pdf', x, y, xperiod, point,name, show=False)

if mode=='best':
    data = [np.load(d + 'info/map_info_structurized.npz', allow_pickle=True) for d in directory]
    K = [df['K'] for df in data]
    assert np.all(np.array_equal(x, K[0]) for x in K)
    K = K[0]
    epu = [df['energy_if_xsp'] for df in data]
    x = K[:,:, 0]
    y = K[:,:, 1]
    energy_diff=epu[1]-epu[0]
    energy_diff[np.isnan(energy_diff)]=2*np.nanmax(energy_diff)

    mc.plot_map(x,y,energy_diff,file=result_directory + 'epu_diff',name='Epu from cone')

    min_state = np.nanargmin(epu, axis=0)
    mc.rect_plot(x, y, min_state)
    plt.savefig(result_directory + 'Best.pdf')
    if show: plt.show()
    plt.close('all')

    if np.all(np.array(x.shape)>1):
        eqx,eqy=points(x,y,energy_diff)
        if len(eqx)>0:
            plt.plot(eqx[:,0],eqx[:,1],'b.')
        if len(eqy) > 0:
            plt.plot(eqy[:, 0], eqy[:, 1], 'b.')
        plt.xlim([x.min(),x.max()])
        plt.ylim([y.min(), y.max()])
        plt.xlabel('$K_{bulk}/D^2$', fontsize=16)
        plt.ylabel('$K_{surf}/D^2$', fontsize=16)
        plt.savefig(result_directory + 'Best_points.pdf')
        if show: plt.show()
        plt.close('all')
        print(f'{eqx = }')
        print(f'{eqy = }')

    min_state = (((epu[0]-epu[1])/np.abs(epu[0]))>0.02).astype('int')
    mc.rect_plot(x, y, min_state)
    plt.savefig(result_directory + 'Bestpm.pdf')
    if show: plt.show()
    plt.close('all')

    if point is not None:
        point_column = np.argmin(np.linalg.norm(x - point[0], axis=1))
        point_row = np.argmin(np.linalg.norm(y - point[1], axis=0))
        print('Point = ', x[point_column][0], y[:, point_row][0])
        axisn = point_column
        z0 = epu[0][axisn, :]
        z1 = epu[1][axisn, :]
        print(x[:, 0])
        print(x[axisn, 0])
        plt.plot(y[axisn, np.invert(np.isnan(z0))], z1[np.invert(np.isnan(z0))], 'bx',label='zcone')
        plt.plot(y[axisn, np.invert(np.isnan(z1))], z1[np.invert(np.isnan(z1))], 'r.',label='xsp')
        plt.title('$Energy per unit, K_{bulk} = ' + '{:.2f}$'.format(x[point_column][0]), fontsize=16)
        plt.xlabel('$K_{surf}/D^2$', fontsize=16)
        plt.ylabel('$Energy$', fontsize=16)
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_directory + '/Energy_pu_s.pdf')
        if show: plt.show()
        plt.close('all')

        z2=z1-z0
        print(x[:, 0])
        print(x[axisn, 0])
        plt.plot(y[axisn, np.invert(np.isnan(z2))], z2[np.invert(np.isnan(z2))], 'r.')
        plt.title('$Energy per unit, K_{bulk} = ' + '{:.2f}$'.format(x[point_column][0]), fontsize=16)
        plt.xlabel('$K_{surf}/D^2$', fontsize=16)
        plt.ylabel('$Energy$', fontsize=16)
        plt.tight_layout()
        plt.savefig(result_directory + '/Energy_diff_pu_s.pdf')
        if show: plt.show()
        plt.close('all')

        axisn = point_row
        z0 = epu[0][:, axisn]
        z1 = epu[1][:, axisn]
        plt.plot(x[np.invert(np.isnan(z0)), axisn], z0[np.invert(np.isnan(z0))], 'bx',label='zcone')
        plt.plot(x[np.invert(np.isnan(z1)), axisn], z1[np.invert(np.isnan(z1))], 'r.',label='xsp')
        plt.title('$Energy per unit, K_{surf} = '+'{:.2f}$'.format(y[:, point_row][0]), fontsize=16)
        plt.xlabel('$K_{bulk}/D^2$', fontsize=16)
        plt.ylabel('$Energy$', fontsize=16)
        plt.tight_layout()
        plt.savefig(result_directory + '/Energy_pu_n.pdf')
        if show: plt.show()
        plt.close('all')

        z2=z1-z0
        plt.plot(x[np.invert(np.isnan(z2)), axisn], z2[np.invert(np.isnan(z2))], 'r.')
        plt.title('$Energy per unit, K_{surf} = ' + '{:.2f}$'.format(y[:, point_row][0]), fontsize=16)
        plt.xlabel('$K_{bulk}/D^2$', fontsize=16)
        plt.ylabel('$Energy$', fontsize=16)
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_directory + '/Energy_diff_pu_n.pdf')
        if show: plt.show()
        plt.close('all')

elif mode=='skt':
    data = [np.load(d + 'info/map_info_structurized.npz', allow_pickle=True) for d in directory]
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
    plt.plot(cone_x[:, 0], cone_x[:, 1], 'k.')
    plt.plot(cone_y[:, 0], cone_y[:, 1], 'k.')
    plt.plot(skyrmon_x[:, 0], skyrmon_x[:, 1], 'r.')
    plt.plot(skyrmon_y[:, 0], skyrmon_y[:, 1], 'r.')
    plt.plot(toron_x[:, 0], toron_x[:, 1], 'g.')
    plt.plot(toron_y[:, 0], toron_y[:, 1], 'g.')
    plt.xlim([x.min(), x.max()])
    plt.ylim([y.min(), y.max()])
    plt.xlabel('$K_{bulk}/D^2$', fontsize=16)
    plt.ylabel('$K_{surf}/D^2$', fontsize=16)
    plt.savefig(result_directory + 'Best_points.pdf')
    if show: plt.show()
    plt.close('all')
