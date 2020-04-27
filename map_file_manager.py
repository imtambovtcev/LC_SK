import os
import sys
import numpy as np
from pathlib import Path

def Kaxis_sort(Kaxis):
    return [np.sort(np.unique(np.array(i))) for i in Kaxis]

def content(directory,dimension=None):
    filelist = [Path(file) for file in os.listdir(directory) if len(file)>3 and Path(file).suffix == '.npz' and len(str.split(file, '_'))>2]
    if len(filelist)>0:
        state_name = np.array([str.split(file.stem, '_')[0] for file in filelist])
        split = np.array([len(str.split(file.stem, '_')) for file in filelist])
        assert np.all(split == split[0])
        if dimension:
            split=dimension
        else:
            split = int(split[0])-1
        K=np.full([len(filelist),split],np.nan)
        for i in range(len(filelist)):
            for j in range(split):
                K[i,j]=float(str.split(filelist[i].stem, '_')[j+1])
        assert np.all(state_name == state_name[0])
        state_name = state_name[0]
        return K,state_name
    else:
        return np.array([]),'void'

def axis(K):
    if K.shape[0] != 0:
        return [np.array(sorted(list(dict.fromkeys(i.tolist())))) for i in K.T]
    else:
        return np.array([])

def missing(K,Kaxis):
    Kaxis = Kaxis_sort(Kaxis)
    full_K=(np.array(np.meshgrid(*Kaxis)).T.reshape(-1, len(Kaxis))).tolist()
    K_list=K.tolist()
    return np.array([f for f in full_K if f not in K_list])

def add(Kaxis,add):
    if len(Kaxis) != 0:
        Kaxis = [np.array(np.array(Kaxis[i]).tolist() + np.array(add[i]).tolist()) for i in range(len(Kaxis))]
    else:
        Kaxis = [np.array(a) for a in add]
    return Kaxis_sort(Kaxis)

def double(Kaxis):
    assert len(Kaxis) != 0
    Kaxis=Kaxis_sort(Kaxis)
    Kaxis=[np.array(i.tolist()+((i[:-1]+i[1:])/2).tolist()) for i in Kaxis]
    Kaxis = Kaxis_sort(Kaxis)
    return Kaxis

def file_manager(directory,params={},dimension=None):
    if 'source' in params:
        K, state_name = content(params['source'],dimension=dimension)
        Kaxis = axis(K)
        Kaxis = Kaxis_sort(Kaxis)
        K, _  = content(directory, dimension=dimension)
    else:
        K, state_name = content(directory,dimension=dimension)
        Kaxis=axis(K)
        Kaxis=Kaxis_sort(Kaxis)
        if 'add' in params:
            Kaxis=add(Kaxis,params['add'])
        if 'double' in params and params['double']:
            Kaxis = double(Kaxis)
    return missing(K,Kaxis),Kaxis

if __name__ == "__main__":
    directory = './' if len(sys.argv) <= 1 else sys.argv[1]
    print(f'{directory}')
    K,state_name=content(directory)
    if K.shape[0]==0:
        print('Folder is empty')
    else:
        Kaxis= axis(K)
        add = missing(K,Kaxis)
        print(f'shape\t min\t max')
        for i in Kaxis:
            print(f'{i.shape}\t{min(i)}\t{max(i)}')

        if add.shape[0]==0:
            print('Map is complete')
        else:
            print(f'{add.shape[0]} points are missing')

        Kaxis=(double(Kaxis))
        print('After double:')
        print(f'shape\t min\t max')
        for i in Kaxis:
            print(f'{i.shape}\t{min(i)}\t{max(i)}')
