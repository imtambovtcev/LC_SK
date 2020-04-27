import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt
import magnes
import map_file_manager as mfm
import double_energy


file='initials/skyrmion.npz'
directory='skyrmion_distance/'
state_name='skyrmion'

'''
file='initials/toron.npz'
directory='toron_distance/'
state_name='toron'
'''

if not os.path.exists(directory):
    os.makedirs(directory)


Np = 50
max_distance = 100


dlist=mfm.file_manager(directory,
					   params={'double':False,
							   'add': [np.round(np.linspace(-max_distance,max_distance,Np),decimals=0).tolist(),np.round(np.linspace(-max_distance,max_distance,Np),decimals=0).tolist()]
							   }
					   )

container = magnes.io.load(file)
ini = container["STATE"]
skyrmion_cut=30
sample_size=350
skyrmion=magnes.utils.state_reshape(ini,[skyrmion_cut,skyrmion_cut,ini.shape[2]],[0.5*(skyrmion_cut-ini.shape[0]),0.5*(skyrmion_cut-ini.shape[1]),0])
ini=magnes.utils.state_reshape(ini,[sample_size,sample_size,ini.shape[2]],[0.5*(sample_size-ini.shape[0]),0.5*(sample_size-ini.shape[1]),0])

system=container.extract_system()
primitives=system.primitives
exchange=system.exchange
representatives=system.representatives
anisotropy=system.anisotropy
bc=system.bc
size=ini.shape[:3]
print(system.size)

system = magnes.System(primitives, representatives, size, bc)
for i in exchange:
    system.add(i)
for i in anisotropy:
    system.add(magnes.Anisotropy(i.strength))
state = system.field3D()
state.upload(ini)
state.satisfy_constrains()

plot = False
maxtime = 2000
alpha = 0.1
precision = 5e-5
catcher = magnes.EveryNthCatcher(1000)
reporters = [magnes.TextStateReporter()]
if plot:
    reporters.append(magnes.graphics.GraphStateReporter())
    reporters.append(magnes.graphics.VectorStateReporter3D(slice2D='xy',sliceN=int(size[2]/2)))
    reporters.append(magnes.graphics.VectorStateReporter3D(slice2D='xz', sliceN=int(size[1] / 2)))
minimizer = magnes.StateGDwM(system, reference=None, stepsize=alpha, maxiter=None, maxtime=maxtime, precision=precision,
                             reporter=magnes.MultiReporter(reporters), catcher=catcher, speedup=5)

minimizer.optimize(state)
ini=state.download()
container = magnes.io.container()
container.store_system(system)
container["STATE"] = ini
container.save(directory +'ini.npz')

t0=time.time()

for idx,d in enumerate(dlist,start=1):
    x=int(d[0]-0.5*skyrmion_cut+0.5*sample_size)
    y=int(d[1] - 0.5 * skyrmion_cut+0.5*sample_size)
    s=np.copy(ini)
    s[x:x+skyrmion_cut,y:y+skyrmion_cut,:,:,:]=skyrmion
    print(s.shape)

    state.upload(s)
    state.satisfy_constrains()
    minimizer.optimize(state)
    s=state.download()
    container = magnes.io.container()
    container.store_system(system)
    container["STATE"] = s
    container.save(directory +state_name+ '_{}_{}.npz'.format(int(d[0]), int(d[1])))
    fig,_,_=magnes.graphics.plot_field3D(system,state,slice2D='xy',sliceN=int(s.shape[2]/2))
    fig.savefig(directory +state_name+ '_{}_{}.pdf'.format(int(d[0]), int(d[1])))
    plt.close('all')
    if idx%10==9:
        print('{} completed out of {}'.format(idx+1, len(dlist)))
        print('Running time {:.0f}s Estimated time {:.0f}s'.format(time.time() - t0,
                                                                   (time.time() - t0) * (len(dlist) - (idx+1)) / (idx+1)))

double_energy.double_energy(directory)