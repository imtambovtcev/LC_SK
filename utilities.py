import numpy as np
from scipy.interpolate import CubicSpline
import magnes
from pathlib import Path

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

