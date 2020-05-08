import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
import matplotlib
from scipy.optimize import curve_fit

font = {'family' : 'sans-serif',
        'size'   : 12}

matplotlib.rc('font', **font)
cmap='viridis'

def plot_z(x,y, directory, state_name,show=False):
    os.environ['MAGNES_BACKEND'] = 'numpy'
    import magnes
    name = str(state_name)
    cmap = plt.get_cmap('jet', len(y))
    for idx,y0 in enumerate(y):
        container = magnes.io.load(directory + name+'_{:.5f}_{:.5f}.npz'.format(x, y0))
        s = container["STATE"]
        s= s[int(s.shape[0]/2),int(s.shape[1]/2),:,0,2]
        s=s.reshape(-1)
        z=np.array(range(len(s)))
        plt.plot(z, s, c=cmap(idx))
    #plt.rc('text', usetex=True)
    norm = matplotlib.colors.Normalize(vmin=y.min(), vmax=y.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('$K_{surf}/D^2$', rotation=270)
    plt.xlabel('$z$', fontsize=16)
    plt.ylabel('$s_z$', fontsize=16)
    plt.tight_layout()
    plt.savefig(directory + 'info/z_projection.pdf')
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
    plt.xlabel('$K_{bulk}/D^2$', fontsize=16)
    plt.ylabel('$K_{surf}/D^2$', fontsize=16)
    plt.colorbar()
    plt.tight_layout()

def plot_map(x,y,z,file,name,show=False,cmap='terrain'):
    if np.all(np.array(z.shape) > 1):
        rect_plot(x, y, z)
        plt.savefig(file + '_r.pdf')
        if show: plt.show()
        plt.close('all')

        plt.contourf(x, y, z)
        plt.xlabel('$K_{bulk}/D^2$', fontsize=16)
        plt.ylabel('$K_{surf}/D^2$', fontsize=16)
        plt.title(name)
        plt.colorbar()

        f = open(file + '_r.txt', "w")
        np.savetxt(f, np.array([x.reshape(-1), y.reshape(-1),z.reshape(-1)]).T, header="x y z")
    else:
        if np.all(x == x[0]):
            plt.plot(np.squeeze(y), np.squeeze(z), 'r.')
            plt.xlabel('$K_{surf}/D^2$', fontsize=16)
            f = open(file + '_r.txt', "w")
            np.savetxt(f, np.array([y.reshape(-1), z.reshape(-1)]).T, header="y z")
        elif np.all(y == y[0]):
            plt.plot(np.squeeze(y), np.squeeze(z), 'r.')
            plt.xlabel('$K_{bulk}/D^2$', fontsize=16)
            f = open(file + '_r.txt', "w")
            np.savetxt(f, np.array([x.reshape(-1), z.reshape(-1)]).T, header="x z")
        plt.ylabel(name, fontsize=16)

    plt.tight_layout()
    plt.savefig(file + '.pdf')
    if show: plt.show()
    plt.close('all')

def point_plot(file,x,y,z,point,show=False,plot_z_projection=False,state_name=False):
    point_column = np.argmin(np.linalg.norm(x - point[0], axis=1))
    point_row = np.argmin(np.linalg.norm(y - point[1], axis=0))
    print('Point = ', x[point_column][0], y[:, point_row][0])
    axisn = point_column
    z1 = z[axisn, :]
    px = y[axisn, np.invert(np.isnan(z1))]
    py = z1[np.invert(np.isnan(z1))]
    if len(py) > 1:
        plt.plot(px, py, 'r.')
        plt.xlabel('$K_{surf}/D^2$', fontsize=16)
        plt.ylabel(file, fontsize=16)
        plt.title('$K_{bulk}/D^2 = $'+'{:.3f}'.format(x[point_column,0]), fontsize=16)
        plt.tight_layout()
        plt.savefig(directory + 'info/' + file + '_p_surf.pdf')
        if show: plt.show()
        plt.close('all')

        int = np.poly1d(np.polyfit(px, py, 5))
        ipx = np.linspace(px.min(), 50, num=100, endpoint=True)
        ipy = int(ipx)

        plt.plot(ipx, ipy, 'r')
        plt.xlabel('$K_{surf}/D^2$', fontsize=16)
        plt.ylabel(file, fontsize=16)
        plt.title('$K_{bulk}/D^2 = $' + '{:.3f}'.format(x[point_column, 0]), fontsize=16)
        plt.tight_layout()
        plt.savefig(directory + 'info/' + file + '_pi_surf.pdf')
        if show: plt.show()
        plt.close('all')

    axisn = point_row
    z2 = z[:, axisn]
    px = x[np.invert(np.isnan(z2)), axisn]
    py = z2[np.invert(np.isnan(z2))]
    if len(py) > 1:
        plt.plot(px, py, 'r.')
        plt.xlabel('$K_{bulk}/D^2$', fontsize=16)
        plt.ylabel(file, fontsize=16)
        plt.title('$K_{surf}/D^2 = $' + '{:.3f}'.format(y[point_row, 0]), fontsize=16)
        plt.tight_layout()
        plt.savefig(directory + 'info/' + file + '_p_bulk.pdf')
        if show: plt.show()
        plt.close('all')

        int = np.poly1d(np.polyfit(px, py, 2))
        ipx = np.linspace(-0.3, px.max(), num=100, endpoint=True)
        ipy = int(ipx)
        plt.plot(ipx, ipy, 'r')
        plt.xlabel('$K_{bulk}/D^2$', fontsize=16)
        plt.ylabel(file, fontsize=16)
        plt.title('$K_{surf}/D^2 = $' + '{:.3f}'.format(y[point_row, 0]), fontsize=16)
        plt.tight_layout()
        plt.savefig(directory + 'info/' + file + '_pi_bulk.pdf')
        if show: plt.show()
        plt.close('all')

    if plot_z_projection:
        print(f'{plot_z_projection = }')
        plot_z(x[point_column][0], y[axisn, np.invert(np.isnan(z1))], directory=directory,
               state_name=state_name, show=False)
        plot_z_projection = False

def plot_cut(x_ini,y_ini,z_ini,directory,name,n=-1,show=False,cmap='terrain'):
    name = str(name)
    if n == -1:
        n0=1
    else:
        n0=int(len(y_ini)/n)+1
    x=x_ini[0::n0,0]
    y=y_ini[0,:]
    z=z_ini[-1::-n0,:]
    cmap = plt.get_cmap('jet', len(x))
    for idx, z0 in enumerate(z):
        plt.plot(y, z0,'.', c=cmap(idx),label='$K_{bulk}/D^2 ='+' {:.3f}$'.format(x[idx]))

        zerror=0.1*np.ones(z0.shape)
        upperlimits = np.ones(z0.shape)
        lowerlimits = np.ones(z0.shape)
        plt.errorbar(y, z0, c=cmap(idx), yerr=zerror, uplims=upperlimits, lolims=lowerlimits)
    # plt.rc('text', usetex=True)
    norm = matplotlib.colors.Normalize(vmin=x.min(), vmax=x.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    #cbar = plt.colorbar(sm)
    #cbar.ax.get_yaxis().labelpad = 15
    #cbar.ax.set_ylabel('$K_{bulk}/D^2$', rotation=270)
    plt.xlabel('$K_{surf}/D^2$', fontsize=16)
    plt.ylabel(name, fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory + 'info/all_{}_surf.pdf'.format(name))
    if show: plt.show()
    plt.close('all')
    if n == -1:
        n0=1
    else:
        n0=int(len(x_ini.T)/n)+1
    x = x_ini[:, 0]
    y = y_ini[0,0::n0]
    z = z_ini.T[0::n0, :]
    cmap = plt.get_cmap('jet', len(y))
    for idx, z0 in enumerate(z):
        plt.plot(x, z0, c=cmap(idx),label='$K_{surf}/D^2 ='+' {:.3f}$'.format(y[idx]))

        zerror = 0.1 * np.ones(z0.shape)
        upperlimits = np.ones(z0.shape)
        lowerlimits = np.ones(z0.shape)
        plt.errorbar(x, z0, c=cmap(idx), yerr=zerror, uplims=upperlimits, lolims=lowerlimits)
    # plt.rc('text', usetex=True)
    norm = matplotlib.colors.Normalize(vmin=y.min(), vmax=y.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    #cbar = plt.colorbar(sm)
    #cbar.ax.get_yaxis().labelpad = 15
    #cbar.ax.set_ylabel('$K_{surf}/D^2$', rotation=270)
    plt.xlabel('$K_{bulk}/D^2$', fontsize=16)
    plt.ylabel(name, fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory + 'info/all_{}_bulk.pdf'.format(name))
    if show: plt.show()
    plt.close('all')


localisation_criteria = 100
def map_color(directory,show=False, point=None,plot_z_projection=False):
    data = np.load(directory +'info/map_info_structurized.npz', allow_pickle=True)

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
                if not (np.all(sk) or np.all(np.invert(sk))):
                    state = np.zeros(c.shape)
                    state[sk] = 1.
                    plt.contour(x,y, state, levels=state_levels, cmap='autumn', hatches=['-', '/', '\\', '//'], extend='both',
                                alpha=0.8)
                if not (np.all(t) or np.all(np.invert(t))):
                    state = np.zeros(c.shape)
                    state[t] = 1.
                    plt.contour(x, y, state, levels=state_levels, cmap='summer', hatches=['-', '/', '\\', '//'], extend='lower',
                                alpha=0.8)
                plt.tight_layout()
                plt.savefig(directory+'info/' + 'state.pdf')
                if show: plt.show()
                plt.close('all')
            elif file == 'K' or file == 'allow_pickle':
                pass
            else:
                try:
                    z = data[file]
                    try:
                        plot_map(x, y, z,file=directory+'info/' + file,name=file,show=show,cmap=cmap)
                        plot_cut(x, y, z,directory=directory,name=file,show=show)
                    except:()
                    if point is not None:
                        point_plot(file,x,y,z,point,show=show,plot_z_projection=plot_z_projection,state_name=data['state_name'])
                except:
                    print(file, 'was not identified')

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