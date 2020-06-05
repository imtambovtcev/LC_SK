from scipy.io import loadmat
import numpy as np
import magnes
import magnes.graphics
import sys

def convert(filename,dubilcate=0,y_size=0,save=True):
    dubilcate=int(dubilcate)
    y_size=int(y_size)
    assert filename[-4:]=='.mat'
    data = loadmat(filename)
    x=np.array(data['Sx'])
    y=np.array(data['Sy'])
    z=np.array(data['Sz'])
    print(f'{x.shape = }')
    print(f'{y.shape = }')
    print(f'{z.shape = }')
    if y_size == 0:
        x=np.moveaxis(x,0,2)
        x=np.moveaxis(x,0,1)
        y=np.moveaxis(y,0,2)
        y=np.moveaxis(y,0,1)
        z=np.moveaxis(z,0,2)
        z=np.moveaxis(z,0,1)
        s = np.zeros([x.shape[0], x.shape[1], x.shape[2], 1, 3])
        s[:, :, :, 0, 0] = x
        s[:, :, :, 0, 1] = y
        s[:, :, :, 0, 2] = z
    else:
        s=np.zeros([x.shape[0],y_size,x.shape[2],1,3])
    print(f'{s.shape = }')
    s=s[:,:,2:-2,:]
    print(f'{s.shape = }')
    s_norm=np.squeeze(np.linalg.norm(s,axis=4))
    assert np.all(np.abs(s_norm-1)<0.01)
    if dubilcate>0:
        s1=np.copy(s)
        for i in range(dubilcate):
            s1=np.concatenate([s1,s],axis=0)
    size=list(s_norm.shape)
    primitives = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
    representatives = [(0., 0., 0.)]
    bc=[magnes.BC.PERIODIC,magnes.BC.PERIODIC,magnes.BC.FREE]
    system = magnes.System(primitives, representatives, size, bc)
    print(system)
    state=system.field3D()
    state.upload(s)
    state.satisfy_constrains()
    s = state.download()
    if save:
        container = magnes.io.container()
        container.store_system(system)
        container["STATE"] = s
        container.save(filename[:-4] + '.npz')
    return s

if __name__ == "__main__":
    convert(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),save=True)