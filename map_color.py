import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
import matplotlib
from scipy.optimize import curve_fit
from pathlib import Path
import magnes
from scipy.interpolate import interp1d

font = {'family' : 'sans-serif',
        'size'   : 12}

matplotlib.rc('font', **font)
cmap='viridis'

def smooth(x,y):
    xs = np.linspace(x.min(), x.max(), 100)
    f = interp1d(x, y, kind='cubic')
    return [xs,f(xs)]

def get_points(x,y,n):
    if isinstance(n,int):
        if x.shape[0]>1:
            xpoints=np.array(range(x.shape[0]))[::int(x.shape[0]/n)]
        else:
            xpoints=[0]
        if x.shape[1]>1:
            ypoints=np.array(range(x.shape[1]))[::int(x.shape[1]/n)]
        else:
            ypoints=[0]
    else:
        xpoints=n[0]
        ypoints=n[1]

    return [np.array(sorted(list(xpoints),key=lambda x: np.abs(x))),np.array(sorted(list(ypoints)))]

def point_arg(x,y, point):
    return [np.argmin(np.linalg.norm(x - point[0], axis=1)),np.argmin(np.linalg.norm(y - point[1], axis=0))]

def plot_z(x,y,point, directory, state_name,n=20,show=False,caption=True,colorbar_type='text'):
    directory = Path(directory)
    name = str(state_name)
    x0,y0=point_arg(x,y,point)
    xpoints,ypoints=get_points(x,y,n)
    #print(f'{xpoints = }\t{ypoints = }')
    cmap = plt.get_cmap('jet', len(xpoints))
    for ct,xn in enumerate(xpoints):
        try:
            container = magnes.io.load(str(directory.joinpath(name+'_{:.5f}_{:.5f}.npz'.format(x[xn,0], y[0,y0]))))
            if 'STATE' in container:
                s = container["STATE"]
            else:
                s = list(container["PATH"])[0]
            l = s.shape[2]
            s= s[int(s.shape[0]/2),int(s.shape[1]/2),:,0,2]
            s=s.reshape(-1)
            z=np.array(range(len(s)))
            if colorbar_type == 'text':
                color = cmap(int((len(xpoints) - 1) * (x[xn, 0] - x[xpoints, 0].min()) / (
                            x[xpoints, 0].max() - x[xpoints, 0].min())))
                plt.plot(z / l, s, c=color, linewidth=2)
                n = int(len(z) * ct / len(xpoints))
                angle = np.arctan((s[n + 1] - s[n]) / (z[n + 1] / l - z[n] / l)) * 180 / np.pi
                trans_angle = plt.gca().transData.transform_angles(np.array((angle,)),
                                                                   np.array([s.min(), s.max()]).reshape((1, 2)))[0]
                plt.text(z[n]/l,s[n]-0.015,str(x[xn,0]),c=color,rotation=trans_angle, rotation_mode='anchor',
                         bbox=dict(boxstyle='round', facecolor='white', edgecolor='none',alpha=0.5))
            else:
                color = cmap(int((len(xpoints) - 1) * (x[xn, 0] - x[xpoints, 0].min()) / (
                            x[xpoints, 0].max() - x[xpoints, 0].min())))
                plt.plot(z / l, s, c=color, linewidth=2)
        except:
            print(str(directory.joinpath(name+'_{:.5f}_{:.5f}.npz'.format(x[xn,0], y[0,y0])))+'\tplot z bulk fail')
    #plt.rc('text', usetex=True)
    norm = matplotlib.colors.Normalize(vmin=x[xpoints,0].min(), vmax=x[xpoints,0].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if colorbar_type=='text':
        print(colorbar_type)
    elif colorbar_type=='colorbar':
        cbar = plt.colorbar(sm)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(r'$\kappa^b$', rotation=270,fontsize=16)
    plt.xlabel(r'$z/l_0$', fontsize=24)
    plt.ylabel(r'$m_z$', fontsize=24)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    if caption:
        plt.title(r'$K^s\; J/D^2 ='+' {:.3f}$'.format(y[0,y0]))
    plt.tight_layout()
    plt.savefig(directory.joinpath('info').joinpath('z_projection_bulk.pdf'))
    if show: plt.show()
    plt.close('all')

    cmap = plt.get_cmap('jet', 256)
    for ct,yn in enumerate(ypoints):
        try:
            container = magnes.io.load(str(directory.joinpath(name + '_{:.5f}_{:.5f}.npz'.format(x[x0,0], y[0,yn]))))
            s = container["STATE"]
            print(f'{s.shape = }')
            l = s.shape[2]
            print(f'{int(s.shape[0] / 2) = }')
            print(f'{int(s.shape[1] / 2) = }')
            s = s[int(s.shape[0] / 2), int(s.shape[1] / 2), :, 0, 2]

            s = s.reshape(-1)
            z = np.array(range(len(s)))
            if colorbar_type == 'text':
                color = cmap(int((256 - 1) * ct/len(ypoints)))
                plt.plot(z / l, s, c=color, linewidth=2)
                n = int(len(z) * ct / len(ypoints))
                angle = np.arctan((s[n + 1] - s[n]) / (z[n + 1] / l - z[n] / l)) * 180 / np.pi
                trans_angle = plt.gca().transData.transform_angles(np.array((angle,)),
                                                                   np.array([0 if s.min()>=0 else -1, 1]).reshape((1, 2)))[0]
                plt.text(z[n]/l,s[n]-0.02,str(y[0,yn]),c=color,rotation=trans_angle, rotation_mode='anchor',
                         bbox=dict(boxstyle='round', facecolor='white', edgecolor='none',alpha=0.9))
            else:
                color = cmap(
                    int((256 - 1) * (y[0, yn] - y[0, ypoints].min()) / (y[0, ypoints].max() - y[0, ypoints].min())))
                plt.plot(z / l, s, c=color, linewidth=2)
        except:
            print(str(directory.joinpath(name + '_{:.5f}_{:.5f}.npz'.format(x[x0,0], y[0,yn]))) + '\t z surf fail')
    # plt.rc('text', usetex=True)
    norm = matplotlib.colors.Normalize(vmin=y[0,ypoints].min(), vmax=y[0,ypoints].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if colorbar_type == 'text':
        print(colorbar_type)


    elif colorbar_type == 'colorbar':
        cbar = plt.colorbar(sm)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(r'$\kappa^b$', rotation=270, fontsize=16)
    plt.xlabel(r'$z/l_0$', fontsize=24)
    plt.ylabel(r'$m_z$', fontsize=24)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    if caption:
        plt.title(r'$K^u\; J/D^2 ='+' {:.3f}$'.format(x[x0, 0]))
    plt.tight_layout()
    plt.savefig(directory.joinpath('info').joinpath('z_projection_surf.pdf'))
    if show: plt.show()
    plt.close('all')

def rect_plot(x,y,z,cmap='terrain'):
    z=np.flipud(z.T)
    xs = (x.max() - x.min())
    ys = (y.max() - y.min())
    xs2 = xs / x.shape[0]
    ys2 = ys /y.shape[1]
    #x=x.reshape(-1)
    #y = y.reshape(-1)
    #z=z.reshape(-1)

    fig, ax = plt.subplots(1)
    #plt.xlim((x.min(),x.max()))
    #plt.ylim((y.min(), y.max()))
    #cs = plt.contourf(x, y, z, levels=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5],cmap='Set3')
    #patches = []
    #for i in range(x.size):
    #    rect = mpatches.Rectangle((x[i]-xs2/2,y[i]-ys2/2), xs2, ys2)
    #    patches.append(rect)
    #collection = PatchCollection(patches, cmap='Set3') #edgecolor=colors[int(z[i])],facecolor=colors[int(z[i])])
    #ax.add_collection(collection)
    #collection.set_array(z/z.max())
    extent = [x.min() - xs2/2, x.max() + xs2/2, y.min() - ys2/2, y.max() + ys2/2]
    plt.imshow(z, interpolation='nearest',cmap=cmap, extent=extent)
    ax.set_aspect(aspect='auto')
    plt.xlabel(r'$\kappa^b$', fontsize=16)
    plt.ylabel(r'$\kappa^s$', fontsize=16)
    plt.colorbar()
    plt.tight_layout()

def plot_map(x,y,z,directory,name,show=False,cmap='terrain'):
    directory = Path(directory)
    if np.all(np.array(z.shape) > 1):
        rect_plot(x, y, z)
        print(str(directory.joinpath('info').joinpath(name+ '_r.pdf')))
        plt.savefig(str(directory.joinpath('info').joinpath(name+ '_r.pdf')))
        if show: plt.show()
        plt.close('all')

        plt.contourf(x, y, z)
        plt.xlabel(r'$\kappa^b$', fontsize=16)
        plt.ylabel(r'$\kappa^s$', fontsize=16)
        plt.title(name)
        plt.colorbar()

        f = open(str(directory.joinpath('info').joinpath(name + '_r.txt')), "w")
        np.savetxt(f, np.array([x.reshape(-1), y.reshape(-1),z.reshape(-1)]).T, header="x y z")
    else:
        if np.all(x == x[0]):
            plt.plot(np.squeeze(y), np.squeeze(z), 'r.')
            plt.xlabel('$\kappa^s$', fontsize=16)
            plt.title('$K^u\; J/D^2 = $' + '{:.3f}'.format(x[0,0]), fontsize=16)
            f = open(str(directory.joinpath('info').joinpath(name+'_r.txt')), "w")
            np.savetxt(f, np.array([y.reshape(-1), z.reshape(-1)]).T, header="y z")
        elif np.all(y == y[0]):
            plt.plot(np.squeeze(x), np.squeeze(z), 'r.')
            plt.xlabel(r'$\kappa^b$', fontsize=16)
            plt.title(r'$K^s\; J/D^2 = $' + '{:.3f}'.format(y[0,0]), fontsize=16)
            f = open(str(directory.joinpath('info').joinpath(name+'_r.txt')), "w")
            np.savetxt(f, np.array([x.reshape(-1), z.reshape(-1)]).T, header="x z")
        plt.ylabel(name, fontsize=16)

    plt.tight_layout()
    plt.savefig(str(directory.joinpath('info').joinpath(name+'.pdf')))
    if show: plt.show()
    plt.close('all')

def plot_point(x,y,z,point,directory, file, show=False,file_for_save=None):
    directory = Path(directory)
    if file_for_save is None:
        file_for_save=file.replace('$','').replace('/','').replace('\\','')
    point_column = np.argmin(np.linalg.norm(x - point[0], axis=1))
    point_row = np.argmin(np.linalg.norm(y - point[1], axis=0))
    print('Point = ', x[point_column][0], y[:, point_row][0])
    axisn = point_column
    z1 = z[axisn, :]
    px = y[axisn, np.invert(np.isnan(z1))]
    py = z1[np.invert(np.isnan(z1))]
    if len(py) > 1:
        plt.plot(px, py, 'r.')
        plt.xlabel(r'$\kappa^s$', fontsize=16)
        plt.ylabel(file, fontsize=16)
        plt.title(r'$K^u\; J/D^2 = $'+'{:.3f}'.format(x[point_column,0]), fontsize=16)
        plt.tight_layout()
        plt.savefig(str(directory.joinpath('info').joinpath(file + '_p_surf.pdf')))
        if show: plt.show()
        plt.close('all')

        int = np.poly1d(np.polyfit(px, py, 5))
        ipx = np.linspace(px.min(), 50, num=100, endpoint=True)
        ipy = int(ipx)

        plt.plot(ipx, ipy, 'r')
        plt.xlabel(r'$\kappa^s$', fontsize=16)
        plt.ylabel(r"{}".format(file), fontsize=16)
        plt.title(r'$K^u\; J/D^2 = $' + '{:.3f}'.format(x[point_column,0]), fontsize=16)
        plt.tight_layout()
        plt.savefig(str(directory.joinpath('info').joinpath(file + '_pi_surf.pdf')))
        if show: plt.show()
        plt.close('all')

    axisn = point_row
    z2 = z[:, axisn]
    px = x[np.invert(np.isnan(z2)), axisn]
    py = z2[np.invert(np.isnan(z2))]
    if len(py) > 1:
        plt.plot(px, py, 'r.')
        plt.xlabel(r'$\kappa^b$', fontsize=16)
        plt.ylabel(r"{}".format(file), fontsize=16)
        plt.title(r'$K^s\; J/D^2 = $' + '{:.3f}'.format(y[0,point_row]), fontsize=16)
        plt.tight_layout()
        plt.savefig(str(directory.joinpath('info').joinpath(file + '_p_bulk.pdf')))
        if show: plt.show()
        plt.close('all')

        int = np.poly1d(np.polyfit(px, py, 2))
        ipx = np.linspace(-0.3, px.max(), num=100, endpoint=True)
        ipy = int(ipx)
        plt.plot(ipx, ipy, 'r')
        plt.xlabel(r'$\kappa^b$', fontsize=16)
        plt.ylabel(r"{}".format(file), fontsize=16)
        plt.title(r'$K^s\; J/D^2 = $' + '{:.3f}'.format(y[0,point_row]), fontsize=16)
        plt.tight_layout()
        plt.savefig(str(directory.joinpath('info').joinpath(file + '_pi_bulk.pdf')))
        if show: plt.show()
        plt.close('all')

def plot_cut(x,y,z,directory,name,n=5,show=False,cmap='terrain',xlim=None,ylim=None,marker=None,line='-',file_for_save=None):
    name = str(name)
    if file_for_save is None:
        file_for_save = name.replace('$', '').replace('/', '').replace('\\', '')
    directory=Path(directory)
    if n == -1:
        xpoints, ypoints = [np.array(range(x.shape[0])),np.array(range(x.shape[1]))]
    else:
        xpoints, ypoints = get_points(x,y,n)
    print(f'{xpoints = }\t{ypoints = }')
    cmap = plt.get_cmap('jet',len(xpoints))
    for idx,xn in enumerate(xpoints):
        plt.plot(y[0,:], z[xn,:],marker=marker,linestyle=line, c=cmap(idx),
                                                 label=r'$K^u\; J/D^2 ='+' {:.3f}$'.format(x[xn,0]))
        # plt.rc('text', usetex=True)
    norm = matplotlib.colors.Normalize(vmin=x[xpoints,0].min(), vmax=x[xpoints,0].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    #cbar = plt.colorbar(sm)
    #cbar.ax.get_yaxis().labelpad = 15
    #cbar.ax.set_ylabel('$\kappa^b$', rotation=270)
    if ylim is not None:
        plt.xlim(ylim)
    plt.xlabel(r'$\kappa^s$', fontsize=16)
    plt.ylabel(r"{}".format(name), fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(directory.joinpath('info').joinpath('all_' +file_for_save+ '_surf.pdf')))
    if show: plt.show()
    plt.close('all')

    cmap = plt.get_cmap('jet', len(ypoints))
    for yn in ypoints:
        plt.plot(x[:,0], z[:,yn],marker=marker,linestyle=line, c=cmap(int((len(ypoints)-1)*(y[0,yn]-y[0,ypoints].min())/(y[0,ypoints].max()-y[ypoints,0].min()))),
                 label=r'$K^s\; J/D^2 ='+' {:.3f}$'.format(y[0,yn]))

    norm = matplotlib.colors.Normalize(vmin=y[0,ypoints].min(), vmax=y[0,ypoints].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    #cbar = plt.colorbar(sm)
    #cbar.ax.get_yaxis().labelpad = 15
    #cbar.ax.set_ylabel('$\kappa^s$', rotation=270)
    if xlim is not None:
        plt.xlim(xlim)
    plt.xlabel(r'$\kappa^b$', fontsize=16)
    plt.ylabel(r"{}".format(name), fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(directory.joinpath('info').joinpath('all_' +file_for_save+ '_bulk.pdf')))
    if show: plt.show()
    plt.close('all')



def plot(x,y,z,ylabel,directory,filename,point=None,show=False,cmap=cmap):
    try:
        plot_map(x, y, z, directory=directory, name=filename, show=show, cmap=cmap)
    except:
        print(ylabel+' plot_map error')
    try:
        plot_cut(x, y, z, directory=directory, name=filename, show=show)
    except:
        print(ylabel+' plot_cut error')
    if point is not None:
        try:
            plot_point(x, y, z, point, directory=directory, file=filename, show=show)
        except:
            print(ylabel + ' plot_point error')




localisation_criteria = 100
def map_color(directory,show=False, point=None, plot_z_projection=True):
    directory=Path(directory)
    data = np.load(str(directory.joinpath('info/map_info_structurized.npz')), allow_pickle=True)
    K = data['K']
    if K.shape[-1] == 3:
        directory = directory.joinpath('best')
        data = np.load(str(directory.joinpath('info/map_info_structurized.npz')), allow_pickle=True)
        K = data['K']
    if K.shape[-1]==2:
        x=K[:,:,0]
        y=K[:,:,1]
        for file in data.files:
            if file == 'state_type':
                map = data[file]
                state_levels = np.array([1])
                c = map == 'cone'
                sk = map == 'skyrmion'
                t = map == 'toron'
                b = map == 'bober'
                if not (np.all(sk) or np.all(np.invert(sk))):
                    state = np.zeros(c.shape)
                    state[sk] = 1.
                    try:
                        plt.contourf(x, y, state, levels=[0,1], cmap='winter', hatches=[None, '-'],extend='both', alpha=0.0)
                    except:()
                    try:
                        plt.contour(x,y, state, levels=state_levels, cmap='autumn', hatches=['-', '/', '\\', '//'], extend='both',alpha=0.8)
                    except:()
                if not (np.all(t) or np.all(np.invert(t))):
                    state = np.zeros(c.shape)
                    state[t] = 1.
                    plt.contourf(x, y, state, levels=[0,1], cmap='winter', hatches=[None, '\\'],extend='both', alpha=0.0)
                    plt.contour(x, y, state, levels=state_levels, cmap='summer', hatches=['-', '/', '\\', '//'], extend='lower',
                                alpha=0.8)
                if not (np.all(b) or np.all(np.invert(b))):
                    state = np.zeros(c.shape)
                    state[b] = 1.
                    try:
                        plt.contourf(x, y, state, levels=[0,1], cmap='winter', hatches=[None, '/'],extend='both', alpha=0.0)
                    except:
                        ()
                    try:
                        plt.contour(x, y, state, levels=state_levels, cmap='winter', hatches=['-', '/', '\\', '//'], extend='lower',
                                alpha=0.8)
                    except:()
                plt.tight_layout()
                plt.savefig(str(directory.joinpath('info/state.pdf')))
                if show: plt.show()
                plt.close('all')
            elif file == 'K' or file == 'allow_pickle' or file == 'state_name':
                pass
            elif file == 'energy_per_unit':
                print('epu')
                epu = data[file]
                plot(x,y,epu,'Energy per unit',directory=directory,filename='epu',point=point,show=show,cmap=cmap)
                if Path(directory).parent.joinpath('ferr').is_dir():
                    try:
                        print('ferr found')
                        data_ferr = np.load(
                            str(Path(directory).parent.joinpath('ferr').joinpath('info/map_info_structurized.npz')),
                            allow_pickle=True)
                        epu_ferr = data_ferr['energy_per_unit']
                        plot(x,y,epu-epu_ferr,'Energy per unit from ferr',directory=directory,filename='epu_f',point=point,show=show,cmap=cmap)
                        print('ferr complete')
                    except:
                        ()
                if Path(directory).parent.joinpath('cone').is_dir():
                    try:
                        print('cone found')
                        data_ferr = np.load(
                            str(Path(directory).parent.joinpath('cone').joinpath('info/map_info_structurized.npz')),
                            allow_pickle=True)
                        epu_ferr = data_ferr['energy_per_unit']
                        plot(x,y,epu-epu_ferr,'Energy per unit from cone',directory=directory,filename='epu_c',point=point,show=show,cmap=cmap)
                        print('cone complete')
                    except:
                        ()
            else:
                try:
                    z = data[file]
                    plot(x,y,z,ylabel=file,directory=directory,filename=file,point=point,show=show,cmap=cmap)
                except:
                    print(file, 'was not identified')

        if point is not None and plot_z_projection:
            try:
                print(f'{plot_z_projection = }')
                plot_z(x,y, point, directory=directory, state_name=data['state_name'],n=5, show=show)
            except:()

if __name__ == "__main__":
    directory = './' if len(sys.argv) <= 1 else sys.argv[1]

    if len(sys.argv) == 1:
        map_color('./',show=False)
    elif len(sys.argv) == 2:
        map_color(sys.argv[1],show=False)
    elif len(sys.argv) == 4:
        map_color(sys.argv[1], show=False,point=[float(sys.argv[2]),float(sys.argv[3])])
    else:
        map_color(sys.argv[1], show=False, point=[float(sys.argv[2]),float(sys.argv[3])],plot_z_projection=sys.argv[4]=='True')