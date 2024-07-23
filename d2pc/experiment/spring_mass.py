import numpy as np
import control as ct
from d2pc.utils import commutation_matrix_sp
from typing import Optional


def spring_mass_vec_fn(ms, ks, ds, num_actuated):
    ls = []
    for i, m in enumerate(ms[:-1]):
        ls.extend([ks[i] / m, ds[i] / m, ks[i+1]/m, ds[i+1]/m])
    ls.extend([ks[-1]/ms[-1], ds[-1]/ms[-1]])
    return np.array(ls + [1/m for m in ms[-num_actuated:]])

def create_spring_mass_sys_ct(ms: np.ndarray, ks: np.ndarray, ds: np.ndarray, num_actuated: int):
    if type(ms) == list:
        ms = np.array(ms)
    elif type(ms) == np.ndarray:
        pass
    else:
        raise ValueError('ms should be a list or np.ndarray')
    assert ms.dtype in [np.float64,
                        np.int64], 'ms should be a list of floats or ints'
    num_masses = len(ms)
    assert num_masses > 2, 'ms should have at least 3 elements'
    assert num_actuated <= num_masses, 'num_actuated should be less than or equal to num_masses'
    assert num_masses == len(ks), 'ms and ks should have the same length'
    assert num_masses == len(ds), 'ms and ds should have the same length'

    ni = 2  # number of local states
    nx = ni * num_masses  # number of overall states
    nth = 4 * (num_masses-1) + 2 + num_actuated
    nu = num_actuated
    nw = num_masses
    if num_masses < 3:
        raise NotImplementedError()
    # Define continuous-time system matrices \dot{x}=A_c*x+B_c*u
    A_c = np.zeros((nx, nx))
    J = np.zeros((nw, nx+nu, nth))

    # i=1: also connected to ground
    A_c[0, :ni] = [0, 1]
    A_c[1, :2*ni] = np.array([-(ks[0]+ks[1]), -
                             (ds[0]+ds[1]), ks[1], ds[1]]) / ms[0]
    J[0, :2, :2] = -np.eye(2)
    J[0, :2, 2:4] = -np.eye(2)
    J[0, 2:4, 2:4] = np.eye(2)

    # 1<i<M-1
    for i in range(1, num_masses-1):
        A_c[i*ni, i*ni+1] = 1
        A_c[i*ni+1, (i-1)*ni:(i+2)*ni] = np.array([ks[i], ds[i], -
                                                   (ks[i]+ks[i+1]), -(ds[i]+ds[i+1]), ks[i+1], ds[i+1]]) / ms[i]
        J[i, (i-1)*ni:i*ni, i*4:i*4+2] = np.eye(2)
        J[i, i*ni:(i+1)*ni, i*4:i*4+2] = -np.eye(2)
        J[i, i*ni:(i+1)*ni, i*4+2:i*4+4] = -np.eye(2)
        J[i, (i+1)*ni:(i+2)*ni, i*4+2:i*4+4] = np.eye(2)

    A_c[-2, -1] = 1
    A_c[-1, -4:] = np.array([ks[-1], ds[-1], -ks[-1], -ds[-1]]) / ms[-1]
    J[-1, nx-4:nx-2, -nu-2:-nu] = np.eye(2)
    J[-1, nx-2:nx, -nu-2:-nu] = -np.eye(2)

    B_c = np.zeros((nx, nu))
    B_c[1-2*nu::2] = np.diag([1/m for m in ms[-nu:]])
    for i in range(1, nu+1):
        J[-i, -i, -i] = 1

    A0 = A_c.copy()
    A0[1::ni] = 0

    P = commutation_matrix_sp(nx+nu, num_masses)
    J = P @ J.reshape(-1, nth)

    return A_c, B_c, J

def create_spring_mass_sys(ms: np.ndarray, ks: np.ndarray, ds: np.ndarray, num_actuated: int, ts: Optional[float] = .1):
    A_c, B_c, J = create_spring_mass_sys_ct(ms, ks, ds, num_actuated)

    num_masses = len(ms)
    ni = 2
    nx = num_masses * ni
    nu = num_actuated

    C = np.eye(nx)[::ni]
    E = np.eye(nx)[:, 1::ni]

    # discretize
    A = np.eye(nx) + A_c * ts
    B = B_c * ts
    th0 = np.hstack([np.eye(nx)[1::2], np.zeros((num_masses, nu))]).ravel("F")

    return A, B, C, E, J*ts, th0
