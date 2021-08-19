import numpy as np
from scipy.interpolate import CubicSpline
import magnes
from pathlib import Path
import minimize
from scipy.interpolate import RegularGridInterpolator

def get_angle(s):
    qm = s[:, int(s.shape[1] / 2), int(s.shape[2] / 2) - 3, 0, 1]
    qp = s[:, int(s.shape[1] / 2), int(s.shape[2] / 2) + 3, 0, 1]
    sp_qm = CubicSpline(range(len(qm)), qm)
    sp_qp = CubicSpline(range(len(qp)), qp)
    roots_qm = sp_qm.derivative().roots().tolist()
    roots_qp = sp_qp.derivative().roots().tolist()
    roots_qm = np.array(
        [i for i in roots_qm if i >= 0 and sp_qm(i) > 0.8 * qm.max() and sp_qm(i) < 1.2 * qm.max()])
    roots_qp = np.array(
        [i for i in roots_qp if i >= 0 and sp_qp(i) > 0.8 * qp.max() and sp_qp(i) < 1.2 * qp.max()])
    idx = 0 if len(roots_qm)==1 else 1 if len(roots_qm)==2 else 2
    return np.abs(np.arctan(6/(roots_qm[idx] - roots_qp[idx]))*360/(2*np.pi))

def get_zperiod(s):
    xz = s[int(s.shape[0] / 2), int(s.shape[1] / 2), :, 0, 0]
    sp_qm = CubicSpline(range(len(xz)), xz)
    roots= np.array(sorted(sp_qm.roots().tolist()))
    roots=roots[np.logical_and(roots>=0,roots<= len(xz))]
    if len(roots)>1:
        dr=np.diff(roots)
        dr=dr[dr>2]
        print(f'{dr = }')
        if len(dr)>0:
            print(f'{100/np.mean(dr) = }')
            return np.mean(dr)
        else:
            return s.shape[2]
    else:
        return s.shape[2]

def get_energy(state):
    s=state.download()
    en=state.energy_contributions_sum()['total'] - s.shape[0] * s.shape[1] * (s.shape[2] - 2) * 3 - s.shape[0] * s.shape[1] * 5
    return en, en / (s.shape[0] * s.shape[1] * s.shape[2])

def state_to_path(file):
    file=Path(file)
    try:
        container = magnes.io.load(str(file))
        if 'STATE' in container:
            container['PATH']=np.array([container['STATE']])
            container.save_npz(str(file))
            print(f'Success {file = }')
    except:
        print(f'Error at {file = }')

def z_invert_state(file):
    file = Path(file)
    container = magnes.io.load(str(file))
    system = container.extract_system()
    s=container['PATH'][0]
    if s[:,:,:,:,2].sum()<0:
        s[:,:,:,:,2]*=-1
    container = magnes.io.container('.npz')
    container.store_system(system)
    container["PATH"] = [s]
    container.save(str(file))

def toplogical_charge(system,s,layer_n):
    layer = s[:, :, layer_n, :, :].reshape([system.size[0], system.size[1], 3])
    dx = np.diff(np.concatenate((layer, layer[-1, :, :].reshape(1, system.size[1], 3)), axis=0), axis=0)
    dy = np.diff(np.concatenate((layer, layer[:, -1, :].reshape(system.size[1], 1, 3)), axis=1), axis=1)
    return 3*np.sum(np.cross(dx,dy)*layer)/(4*np.pi*np.pi)

def toplogical_charge_vertical(system,s,layer_n):
    layer = s[layer_n, :, :, :, :].reshape([system.size[1], system.size[2], 3])
    dx = np.diff(np.concatenate((layer, layer[-1, :, :].reshape(1, system.size[2], 3)), axis=0), axis=0)
    dy = np.diff(np.concatenate((layer, layer[:, -1, :].reshape(system.size[1], 1, 3)), axis=1), axis=1)
    return 3*np.sum(np.cross(dx,dy)*layer)/(4*np.pi*np.pi)

def reshape_vector(vector,new_size):
    print(vector.shape)
    if len(vector)>new_size[2]:
        while len(vector)>new_size[2]:
            vector=np.delete(vector,int(len(vector)/2))
    if len(vector)<new_size[2]:
        while len(vector)<new_size[2]:
            vector=np.insert(vector,int(len(vector)/2),vector[int(len(vector)/2)])
    return vector

def reshape_anisotropy(ani,new_size):
    ani.strength=reshape_vector(ani.strength[0,0,:,0],new_size=new_size).reshape([1,1,new_size[2],1])
    return ani

def reshape_field(field,new_size):
    if np.linalg.norm(field)!=0:
        field = field
    return field

def system_reshape(system,new_size):
    primitives = system.primitives
    representatives = system.representatives
    bc = system.bc
    exchange = system.exchange
    anisotropy = system.anisotropy
    H = system.field

    new_system = magnes.System(primitives, representatives,new_size, bc)
    for ex in exchange:
       new_system.add(ex)
    new_system.set_external_field(reshape_field(H,new_size))
    for ani in anisotropy:
        new_ani=reshape_anisotropy(ani,new_size)
        new_system.add(magnes.Anisotropy(axis=new_ani.axis, strength=new_ani.strength))
    return new_system

def smooth_file_reshape(file,save,new_size):
    file = Path(file)
    container = magnes.io.load(str(file))
    system = container.extract_system()
    s = container['PATH'][0]
    size=s.shape[:3]
    if size[0]!=new_size[0]:
        print('x_size_change')
        s = minimize.change_x_shape(state=s, xsize=new_size[0])
    if size[1] != new_size[1]:
        print('y_size_change')
        s = minimize.change_y_shape(state=s, ysize=new_size[1])
    if size[2]!=new_size[2]:
        print('z_size_change')
        s=minimize.change_z_shape(state=s,zsize=new_size[2])
    if np.all(size==new_size):
        print('already reshaped')
    container = magnes.io.container('.npz')
    container.store_system(system_reshape(system,new_size=new_size))
    container["PATH"] = [s]
    container.save(str(save))

def interpolation(si,new_size):
    #print(f'{si.shape = }')
    old_size=si.shape[:3]
    #print('---interpolation---')
    #print(f'{old_size = }')
    #print(f'{new_size = }')
    x_old = np.linspace(0.,1., old_size[0])
    y_old = np.linspace(0.,1., old_size[1])
    z_old = np.linspace(0.,1., old_size[2])
    #print(f'{x_old.shape = }')
    x_new = np.linspace(0., 1., new_size[0])
    y_new = np.linspace(0., 1., new_size[1])
    z_new = np.linspace(0., 1., new_size[2])
    #print(f'{x_new.shape = }')
    new_grid=np.array(np.meshgrid(x_new,y_new,z_new,indexing='ij'))
    new_grid=np.moveaxis(new_grid,0,-1)
    new_grid = new_grid.reshape([-1, 3])
    #print(f'{new_grid.shape = }')
    interpolating_function = RegularGridInterpolator((x_old, y_old, z_old), si)
    new_si=interpolating_function(new_grid)
    #print(f'{new_si.shape = }')
    new_si=new_si.reshape(new_size)
    #print(f'{new_si.shape = }')
    return new_si

def smooth_state_reshape(s,new_size):
    size=list(s.shape[:3])
    #print(f'0.{size = }')
    sx=interpolation(s[:,:,:,0,0],new_size)
    sy=interpolation(s[:,:,:,0,1],new_size)
    sz=interpolation(s[:,:,:,0,2],new_size)
    s=np.array([sx,sy,sz])
    #print(f'1.{s.shape = }')
    s=np.moveaxis(s,0,-1)
    #print(f'2.{s.shape = }')
    s=np.expand_dims(s,axis=3)
    #print(f'3.{s.shape = }')
    return s

def smooth_path_reshape(path,new_size):
    new_path=np.array([smooth_state_reshape(s,new_size=new_size) for s in path])
    #print(f'{new_path.shape = }')
    return new_path

def get_size(file):
    file = Path(file)
    container = magnes.io.load(str(file))
    s = container['PATH'][0]
    return s.shape[:3]

def get_J(file):
    file = Path(file)
    container = magnes.io.load(str(file))
    system=container.extract_system()
    return system.exchange[0]

def get_D(file):
    file = Path(file)
    container = magnes.io.load(str(file))
    system=container.extract_system()
    return system.exchange[0]