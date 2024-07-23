from scipy.stats import chi2
import numpy as np
import cvxpy as cp
from typing import Optional
from d2pc.system import LTISystem
from d2pc.utils import compute_JDelta, block_diag_cvxpy, dpp_kron
from d2pc.control.dynof_ctrl import DynOFController
import scipy.linalg as la


def constraint_lpv_fixed_ctrlr(sys: LTISystem, covar_pre: cp.Variable, covar_post: cp.Variable,
                               J: np.ndarray, hess_th: np.ndarray,
                               delta: float, dynof: Optional[DynOFController] = None,
                               D: Optional[np.ndarray] = None):
    """
        Constructs LMI constraint for: A(theta) covar_pre A(theta)^T \preceq covar_post \forall theta in \Theta_\delta
        - sys: LTISystem - system
        - covar_pre: cp.Variable (nx, nx) - pre-covariance matrix
        - covar_post: cp.Variable (nx, nx) - post-covariance matrix
        - J: np.ndarray (nw*(nx+nu), nth) - affine parametrization matrix
        - hess_th: np.ndarray (nth, nth) - Hessian of the uncertainty
        - delta: float - confidence level
        - dynof: DynOFController - dynamic output-feedback controller, if not provided will parametrize the controller
        - D: np.ndarray (nth, nth) - shape matrix for the approximate uncertainty set
                                     if provided the approximate uncertainty set is used
    """

    nx, ny, nu, nw = sys.nx, sys.ny, sys.nu, sys.nw

    # params
    if dynof is None:
        Ac = cp.Parameter((nx, nx))
        K = cp.Parameter((nu, nx))
        L = cp.Parameter((nx, ny))
    else:
        Ac, K, L = dynof.Ac, dynof.K, dynof.L
    calA = cp.bmat([[sys.A, sys.B@K], [L@sys.C, Ac]])

    # matrices
    Cq = block_diag_cvxpy(np.eye(sys.nx), K)
    cdelta = chi2.ppf(delta, hess_th.shape[0])
    if D is None:
        Lambda_cp = cp.Variable((nw, nw), PSD=True)
        P_up = dpp_kron(Lambda_cp, hess_th / cdelta)
        P_low = Lambda_cp
        Cq = compute_JDelta(J, nw) @ Cq
    else:
        Lambda_cp = cp.Variable(pos=True)
        P_up = D / cdelta * Lambda_cp
        P_low = np.eye(nw) * Lambda_cp

    Bp = np.vstack([sys.E, np.zeros((nx, nw))])
    nwnth = Cq.shape[0]
    r1 = [-covar_post, np.zeros((2*nx, nwnth)), calA @ covar_pre, Bp @ P_low]
    r2 = [r1[1].T, -P_up, Cq @ covar_pre, np.zeros((nwnth, sys.nw))]
    r3 = [r1[2].T, r2[2].T, -covar_pre, np.zeros((2*nx, sys.nw))]
    r4 = [r1[3].T, r2[3].T, r3[3].T, -P_low]
    stab_cond = cp.bmat([r1, r2, r3, r4])
    return stab_cond, Lambda_cp, (Ac, K, L)


def optimize_D(J: np.ndarray, covar_th: np.ndarray, nw: int):
    """
        Optimize the shape matrix D for the approximate uncertainty set.
        (E^\dagger [A(\vartheta) B(\vartheta)]) D \star \preceq I \forall \vartheta \in \Theta
        J: np.ndarray (nw*(nx+nu), nth) - affine parametrization matrix
        covar_th: np.ndarray (nth, nth) - covariance matrix
        nw: int - disturbance dimension
    """
    d = J.shape[0] // nw
    D_cp = cp.Variable((d, d), PSD=True)

    covar_th_sqrt = la.sqrtm(covar_th)
    M = covar_th_sqrt @ J.T @ dpp_kron(D_cp, np.eye(nw)) @ J @ covar_th_sqrt
    prob = cp.Problem(cp.Minimize(cp.lambda_max(M)), [M >> np.eye(M.shape[0])])
    o = prob.solve()
    return D_cp.value / o
