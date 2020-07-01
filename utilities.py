import numpy as np
from scipy.interpolate import CubicSpline

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
