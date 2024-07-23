import numpy as np
from d2pc.utils import rand_psd, hankelize, setup_ls_arx, block_diag
from d2pc.system import LTISystem
from .str_em_runner import StructuredEMRunner
from .est_setup import EstimationSetup
import scipy.linalg as la
from typing import Optional

# The functions below are used to estimate ARX systems using the EM algorithm
# We use a particular structure to fit ARX models, which is not the standard state-space representation
# This requires input signal to be hankelized before fitting the model
# The ARX model is then converted to a standard state-space model

def est_arx_setup(order: int, ny: int, nu: int,
                  Q: Optional[np.ndarray] = None, Qtype: Optional[str] = "full",
                  R: Optional[np.ndarray] = None, Rtype: Optional[str] = "full",
                  mu0: Optional[np.ndarray] = None, P0: Optional[np.ndarray] = None,
                  force_stable_A: bool = True):
    """
    Create an estimation setup for ARX systems
        - order: int order of the ARX system
        - ny: int number of outputs
        - nu: int number of inputs
        - Q: np.ndarray (ny, ny) initial Q matrix
        - Qtype: str type of Q matrix to be estimated ["full", "scaled", "fixed"]
        - R: np.ndarray (ny, ny) initial R matrix
        - Rtype: str type of R matrix to be estimated ["full", "scaled", "fixed"]
        - mu0: np.ndarray (nx,) initial mu0 vector
        - P0: np.ndarray (nx, nx) initial P0 matrix
        - force_stable_A: bool whether to force the A matrix to be stable, 
                          (advised otherwise kf may blow up in the first iter for long sequences)
    """
    nx = order * ny
    nth = ny * order * (ny + nu)
    J = np.eye(nth)
    th0 = np.zeros(nth)

    A_fixed = np.zeros((nx, nx))
    A_fixed[:-ny, ny:] = np.eye((order-1)*ny)
    AB_fixed = np.hstack([A_fixed, np.zeros((nx, order*nu))])
    E = np.vstack([np.zeros(((order-1)*ny, ny)), np.eye(ny)])

    while True:
        rand_vec = np.random.rand(nth) * .4 - .2
        AB = E @ (J@rand_vec).reshape((ny, nx+order*nu), order="f") + AB_fixed
        A = AB[:, :nx]
        if not force_stable_A or np.max(np.abs(la.eigvals(A))) < 1:
            break
    B = AB[:, nx:]

    C = np.zeros((ny, nx))
    C[:, -ny:] = np.eye(ny)

    mu0 = np.random.rand(nx) if mu0 is None else mu0
    P0 = rand_psd(nx) if P0 is None else P0

    Q = rand_psd(ny) if Q is None else Q
    Qprojectors = [np.eye(ny)]
    Qtypes = [Qtype]
    R = rand_psd(ny) if R is None else R
    Rprojectors = [np.eye(ny)]
    Rtypes = [Rtype]
    sys = LTISystem(A=A, B=B, C=C, E=E, Q=Q, R=R, mu0=mu0, P0=P0)
    return EstimationSetup(sys, Qprojectors, Qtypes, Rprojectors, Rtypes, J, th0)

def convert_arx(arx_sys):
    """
        Convert an ARX system from estimation form to standard form
    """
    nx, ny = arx_sys.nx, arx_sys.ny
    order = nx // ny
    nu = arx_sys.nu // order

    Au = np.block([[np.zeros(((order-2)*nu, nu)),
                  np.eye((order-2)*nu)], [np.zeros((nu, (order-1)*nu))]])
    Ay_fix = np.hstack([np.zeros(((order-1)*ny, ny)), np.eye((order-1)*ny)])
    A = block_diag(Au, Ay_fix)
    A = np.vstack([A, np.hstack([arx_sys.B[:, :-nu], arx_sys.A])[-ny:]])
    B = np.vstack([np.zeros(((order-2)*nu, nu)), np.eye(nu),
                  np.zeros(((order-1)*ny, nu)), arx_sys.B[-ny:, -nu:]])
    C = np.hstack([np.zeros((ny, (order-1)*nu)), arx_sys.C])
    E = np.vstack([np.zeros(((order-1)*nu, ny)), arx_sys.E])
    mu0 = np.concatenate([np.zeros((order-1)*nu), arx_sys.mu0])
    P0 = block_diag(np.zeros(((order-1)*nu, (order-1)*nu)), arx_sys.P0)
    return LTISystem(A, B, C, E, arx_sys.Q, arx_sys.R, mu0, P0)


def estimate_arx_sys(ys: np.ndarray, us: np.ndarray, order: int,
                     Q: Optional[np.ndarray] = None, Qtype: Optional[str] = "full",
                     R: Optional[np.ndarray] = None, Rtype: Optional[str] = "full",
                     mu0: Optional[np.ndarray] = None, P0: Optional[np.ndarray] = None, 
                     warm_start_ls: bool = True, update_init_dist:bool=False, ignore_Theta:bool=False, 
                     convert_to_std=True, **em_kwargs):
    """
    Estimate an ARX system using the EM algorithm
        - ys: np.ndarray (T, ny) output data
        - us: np.ndarray (T, nu) input data
        - order: int order of the ARX system
        - Q: np.ndarray (ny, ny) initial Q matrix
        - Qtype: str type of Q matrix to be estimated ["full", "scaled", "fixed"]
        - R: np.ndarray (ny, ny) initial R matrix
        - Rtype: str type of R matrix to be estimated ["full", "scaled", "fixed"]
        - mu0: np.ndarray (nx,) initial mu0 vector
        - P0: np.ndarray (nx, nx) initial P0 matrix
        - warm_start_ls: bool whether to warm start the ARX system with LS
        - update_init_dist: bool whether to update the initial distribution
        - ignore_Theta: bool whether to ignore the parameter set in the EM algorithm
        - em_kwargs: dict additional arguments for the EM algorithm
    """

    ny, nu = ys.shape[1], us.shape[1]
    setup = est_arx_setup(order, ny, nu, Q=Q, Qtype=Qtype, R=R, Rtype=Rtype, mu0=mu0, P0=P0, force_stable_A=True)

    if warm_start_ls:
        reg, label = setup_ls_arx(order, ys, us)
        th = la.lstsq(reg, label, cond=-1)[0].T

        setup.sys.A[-ny:] = th[:, :order*ny]
        setup.sys.B[-ny:] = th[:, order*ny:]

    opt_problem = StructuredEMRunner(setup, update_init_dist=update_init_dist, ignore_Theta=ignore_Theta)
    us_hank = hankelize(np.vstack([np.zeros((order-1, nu)), us]), order)
    syss, Es = opt_problem.em(ys, us_hank, **em_kwargs)
    if "path" in em_kwargs and em_kwargs["path"]:
        if convert_to_std:
            syss = [convert_arx(sys) for sys in syss]
        return syss, Es
    else:
        if convert_to_std:
            syss = convert_arx(syss)
        return syss, Es
