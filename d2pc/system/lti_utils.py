import numpy as np
import control as ct
from d2pc.system.lti_sys import LTISystem
from d2pc.utils import rand_psd, block_diag, commutation_matrix_sp
import scipy.linalg as la

def generate_AB(nx, nu):
    sys = ct.drss(nx, 1, nu)
    return sys.A, sys.B

def generate_ABC(nx, ny, nu):
    sys = ct.drss(nx, ny, nu)
    return sys.A, sys.B, sys.C

def generate_random_lti(nx, ny, nu, nw=None, minQ=1e-4, minR=1e-4, minP0=1e-4):
    A, B, C = generate_ABC(nx, ny, nu)
    mu0 = np.random.rand(nx)

    nw = nx if nw is None else nw
    E = np.vstack([np.zeros((nx-nw, nw)), np.eye(nw)])
    Q = rand_psd(nw, minQ)
    R = rand_psd(ny, minR)
    P0 = rand_psd(nx, minP0)

    return LTISystem(A=A, B=B, C=C, E=E, Q=Q, R=R, mu0=mu0, P0=P0)