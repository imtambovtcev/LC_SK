from scipy.io import loadmat
import numpy as np
import magnes
import magnes.graphics
import sys
import scipy

def convert(filename):
    assert filename[-4:]=='.npz'
    container = magnes.io.load(filename)
    s = container["STATE"]
    s=np.squeeze(s)
    assert np.all(np.abs(np.linalg.norm(s, axis=3) - 1 < 0.01))
    print(f'{s.shape = }')
    s=np.concatenate([np.zeros([s.shape[0],s.shape[1],2,3]),s,np.zeros([s.shape[0],s.shape[1],2,3])],axis=2)
    print(f'{s.shape = }')
    s = np.moveaxis(s, 2, 0)
    s = np.moveaxis(s, 2, 1)
    sx=s[:,:,:,0]
    sy = s[:, :, :, 1]
    sz = s[:, :, :, 2]
    print(f'{sx.shape = }')
    print(f'{sy.shape = }')
    print(f'{sz.shape = }')
    scipy.io.savemat(filename[:-4]+'.mat', {'Sx':sx,'Sy':sy,'Sz':sz}, appendmat=True, format='5', long_field_names=False, do_compression=False,
                     oned_as='row')

if __name__ == "__main__":
    convert(sys.argv[1])