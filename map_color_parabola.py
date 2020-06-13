import numpy as np
import sys
import os
from pathlib import Path
from minimize import parabola

import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
import matplotlib

font = {'family' : 'sans-serif',
        'size'   : 12}

matplotlib.rc('font', **font)
cmap='viridis'

def period_plot(energy,Kbulk, Ksurf,pb=None,wrong_energy=[]):
    energy=np.array(energy)
    print(f'\t{energy[0,0] = }\t{energy[-1, 0] = }')
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

def map_color_3d(directory,point):
        data = np.load(os.path.join(directory,'info/map_info_structurized.npz'), allow_pickle=True)

        point=np.array(point)

        K = data['K']
        energy=data['energy_per_unit']
        angle=data['angle']
        Kf=K[:,:,0,:2]

        point_column = np.argmin(np.linalg.norm(Kf[:,:,0] - point[0], axis=1))
        point_row = np.argmin(np.linalg.norm(Kf[:,:,1] - point[1], axis=0))

        print(f'{point = }\t{point_column = }\t{point_row = }\t{Kf[point_column,point_row] = }')
        x=K[point_column,point_row,:,2]
        energy=energy[point_column,point_row,:]
        angle=angle[point_column,point_row,:]
        minenergy=np.nanmin(energy)
        minperiod=x[np.nanargmin(energy)]
        print(f'{minperiod = }±{x[1]-x[0]:.1f}\t{minenergy = }')

        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Period')
        ax1.set_ylabel('Energy per unit', color=color)
        ax1.plot(x,energy,'.', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('angle', color=color)  # we already handled the x-label with ax1
        ax2.plot(x, angle,'.', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('$K_{bulk}/D^2 = $'+'{:.5f}, '.format(Kf[point_column,point_row][0])+'$K_{surf}/D^2 = $'+'{:.5f}, '.format(Kf[point_column,point_row][1]))
        plt.tight_layout()
        plt.savefig(os.path.join(directory,'info/energy_{:.5f}_{:.5f}.pdf'.format(*(Kf[point_column,point_row].tolist()))))
        plt.show()

if __name__ == "__main__":
    directory = Path('./') if len(sys.argv) <= 1 else sys.argv[1]
    map_color_3d(sys.argv[1],[float(sys.argv[2]),float(sys.argv[3])] )