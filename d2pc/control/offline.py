import numpy as np
import scipy.linalg as la
from d2pc.utils import *
import cvxpy as cp
from scipy.stats import chi2, norm
from d2pc.system import LTISystem
import control as ct
from typing import Optional
from .dynof_ctrl import DynOFController
from .utils import constraint_lpv_fixed_ctrlr

def _nom_tube_design(sys: LTISystem, dynof: DynOFController, H: np.ndarray, J: np.ndarray, hess_th: np.ndarray, delta: float):
    """
        Synthesize nominal tube shape for the given contraction rate.
        H: np.ndarray (nh, nx+nu) - state-input constraints
        J: np.ndarray (nw*(nx+nu), nth) - affine parametrization matrix
        hess_th: np.ndarray (nth, nth) - Hessian of the parametric uncertainty
        delta: float - probability that the true system is covered
    """
    nx, nw, nu = sys.nx, sys.nw, sys.nu

    # params
    rho_sq_cp = cp.Parameter(pos=True)
    rho_cp = cp.Parameter(pos=True)

    # vars
    calXp = cp.Variable((2*nx, 2*nx), PSD=True)
    gamma = cp.Variable(H.shape[0])

    covar_pre = calXp
    covar_post = rho_sq_cp*calXp

    stab_cond, _, _ = constraint_lpv_fixed_ctrlr(
        sys, covar_pre, covar_post, J, hess_th, delta, dynof=dynof)
    constraints = [stab_cond << 0]

    # constr
    cov_th = la.inv(hess_th)
    cov_th_sqrt = la.sqrtm(cov_th)
    d_sqrt = np.sqrt(chi2.ppf(delta, hess_th.shape[0]))
    Bp = np.vstack([sys.E, np.zeros((nx, nw))])
    IK = block_diag(np.eye(nx), dynof.K)
    mat = d_sqrt * cov_th_sqrt @ J.T @ np.kron(IK, Bp.T)
    c = rho_sq_cp - 2 * rho_cp + 1
    inf_horizon_scale_cond = cp.bmat([[c*np.eye(J.shape[1]), mat], 
                                      [mat.T, dpp_kron(np.eye(2*nx), calXp)]])
    constraints.append(inf_horizon_scale_cond >> 0)

    # objective & related constraints
    IK = block_diag(np.eye(nx), dynof.K)
    for i, h in enumerate(H):
        v = calXp @ IK.T @ h
        constraints.append(
            cp.bmat([[calXp, v[:, None]], [v[None, :], gamma[i][None, None]]]) >> 0)
    obj = cp.Minimize(cp.sum(gamma))
    prob = cp.Problem(obj, constraints)

    def solve(rho):
        rho_sq_cp.value = rho**2
        rho_cp.value = rho
        try:
            obj = prob.solve()
        except cp.SolverError:
            return None
        if prob.status == cp.OPTIMAL:
            return obj, la.inv(calXp.value)
    return solve


def nom_tube_design(sys: LTISystem, dynof: DynOFController, H: np.ndarray, J: np.ndarray,
                    hess_th: np.ndarray, delta: float, rho_ds: Optional[float] = 1e-2):
    """
        Synthesize nominal tube shape.
        sys: LTISystem - system
        dynof: DynOFController - dynamic output-feedback controller
        H: np.ndarray (nh, nx+nu) - state-input constraints
        J: np.ndarray (nw*(nx+nu), nth) - affine parametrization matrix
        hess_th: np.ndarray (nth, nth) - Hessian of the uncertainty
        delta: float - probability that the true system is covered
        rho_ds: float - step size for rho
    """
    prob_tube = _nom_tube_design(sys, dynof, H, J, hess_th, delta)
    ls = []
    for rho in reversed(np.arange(rho_ds, 1, rho_ds)):
        ret = prob_tube(rho)
        if ret is not None:
            obj, calP = ret
            ls.append((obj, rho, calP))
        else:
            break
    if len(ls) == 0:
        raise Exception("No feasible solution found for nominal tube design")
    objs, _, calPs = zip(*ls)
    selected_idx = np.argmin(objs)
    obj, calP = objs[selected_idx], calPs[selected_idx]
    rho = get_contraction_rate(sys, calP, dynof, J, hess_th, delta)
    return obj, rho, calP

def get_contraction_rate(sys: LTISystem, calP: np.ndarray, dynof: DynOFController, J: np.ndarray, hess_th: np.ndarray, delta: float):
    """
        Compute the contraction rate for the given tube shape.
        sys: LTISystem - system
        calP: np.ndarray (nx, nx) - tube shape
        dynof: DynOFController - dynamic output-feedback controller
        J: np.ndarray (nw*(nx+nu), nth) - affine parametrization matrix
        hess_th: np.ndarray (nth, nth) - Hessian of the uncertainty
        delta: float - probability that the true system is covered
    """
    rho_sq_cp = cp.Variable(pos=True)

    calXp = la.inv(calP)
    covar_pre = calXp
    covar_post = rho_sq_cp*calXp

    stab_cond, _, _ = constraint_lpv_fixed_ctrlr(
        sys, covar_pre, covar_post, J, hess_th, delta, dynof=dynof)
    obj = cp.Minimize(rho_sq_cp)
    prob = cp.Problem(obj, [stab_cond << 0])
    prob.solve()
    return np.sqrt(rho_sq_cp.value)


def stoch_tube_design_time_varying(sys: LTISystem, dynof: DynOFController, num_ts: int,
                                   init_xi_covar: np.ndarray, H: np.ndarray,
                                   J: np.ndarray, hess_th: np.ndarray, delta: float):
    """
        Synthesize time-varying error covariance bounds.
        sys: LTISystem - system
        dynof: DynOFController - dynamic output-feedback controller
        num_ts: int - number of time steps
        init_xi_covar: np.ndarray (2*nx, 2*nx) - initial error covariance
        H: np.ndarray (nh, nx+nu) - state-input constraints
        J: np.ndarray (nw*(nx+nu), nth) - affine parametrization matrix
        hess_th: np.ndarray (nth, nth) - Hessian of the uncertainty
        delta: float - probability that the true system is covered
    """
    nx = sys.nx
    # vars
    calXs = [cp.Variable((2*nx, 2*nx), PSD=True) for _ in range(num_ts)]
    gamma = cp.Variable((num_ts-1, H.shape[0]))

    # matrices
    BdBdT = block_diag(sys.E@sys.Q@sys.E.T, dynof.L @ sys.R @ dynof.L.T)
    constraints = []
    if num_ts != 1:
        constraints.append(calXs[0] == init_xi_covar)
    for t, (calX_pre, calX_post) in enumerate(zip(calXs[:-1] + [calXs[-1]], calXs[1:] + [calXs[-1]])):
        covar_pre = calX_pre
        covar_post = calX_post - BdBdT

        stab_cond, _, _ = constraint_lpv_fixed_ctrlr(
            sys, covar_pre, covar_post, J, hess_th, delta, dynof=dynof)
        constraints.append(stab_cond << 0)
        if t != 0:
            # objective & related constraints
            IK = block_diag(np.eye(nx), dynof.K)
            for i, h in enumerate(H):
                v = calX_pre @ IK.T @ h
                constraints.append(
                    cp.bmat([[calX_pre, v[:, None]], [v[None, :], gamma[t-1, i][None, None]]]) >> 0)

    obj = cp.Minimize(cp.sum(gamma))
    prob = cp.Problem(obj, constraints)
    obj = prob.solve()
    calXs = [calX.value for calX in calXs]
    if prob.status != cp.OPTIMAL:
        raise Exception("Stochastic error tube design failed!")
    return obj, np.array(calXs)

def compute_stoch_tight(sigma: np.ndarray, K: np.ndarray, H: np.ndarray, p_cc: float):
    """
        Compute the stochastic tightenging terms given the error covariance bound.
        sigma: np.ndarray (nx, nx) - error covariance bound
        K: np.ndarray (nu, nx) - feedback gain
        H: np.ndarray (nh, nx+nu) - state-input constraints
        p_cc: float - chance constraint satisfaction probability
    """
    IK = block_diag(np.eye(K.shape[1]), K)
    IKS = IK @ sigma @ IK.T
    c = norm.ppf(p_cc)
    return c * np.sqrt([h.T @ IKS @ h for h in H])

def compute_nom_tight(calP: np.ndarray, K: np.ndarray, H: np.ndarray):
    """
        Compute the nominal tightening terms.
        calP: np.ndarray (nx, nx) - nominal error covariance bound
        K: np.ndarray (nu, nx) - feedback gain
        H: np.ndarray (nh, nx+nu) - state-input constraints
    """
    IK = block_diag(np.eye(K.shape[1]), K)
    IKPinv = IK @ la.inv(calP) @ IK.T
    return np.sqrt([h.T @ IKPinv @ h for h in H])


def compute_Sc(dynof: DynOFController, costQ: np.ndarray, costR: np.ndarray):
    """
        Compute the terminal cost weight.
        dynof: DynOFController - dynamic output-feedback controller
        costQ: np.ndarray (nx, nx) - state cost
        costR: np.ndarray (nu, nu) - input cost
    """
    costXi = block_diag(costQ, dynof.K.T @ costR @ dynof.K)
    return ct.dlyap(dynof.calA.T, costXi)


def compute_underbar_c(stoch_tight_tv: np.ndarray, nom_tight: np.ndarray):
    """
        Compute the underbar c value. 
        stoch_tight_tv: np.ndarray (T_err, nh) - time-varying stochastic tightening terms
        nom_tight: np.ndarray (nh,) - nominal tightening terms
    """
    stoch_tight = stoch_tight_tv.max(axis=0)
    return np.min((1 - stoch_tight) / nom_tight)


def compute_bar_sigma(sys: LTISystem, J: np.ndarray, covar_th: np.ndarray, calP: np.ndarray, K: np.ndarray):
    """
        Compute the bar sigma value.
        sys: LTISystem - system
        J: np.ndarray (nw*(nx+nu), nth) - affine parametrization matrix
        covar_th: np.ndarray (nth, nth) - uncertainty covariance
        calP: np.ndarray (nx, nx) - tube shape
        K: np.ndarray (nu, nx) - feedback gain
    """
    nx, nu, nw = sys.nx, sys.nu, sys.nw
    covar_th_sqrt = la.sqrtm(covar_th)
    calP_sqrt = la.sqrtm(calP)
    Bp = np.vstack([sys.E, np.zeros((nx, nw))])
    PsqrtBp = calP_sqrt @ Bp
    Sigma_J_sqrt = np.kron(np.eye(nx+nu), PsqrtBp) @ J @ covar_th_sqrt

    Sigma_J = Sigma_J_sqrt @ Sigma_J_sqrt.T
    I = np.eye(2*nx)
    Ikes = [np.kron(np.eye(nx+nu), I[i][:, None]) for i in range(2*nx)]
    Sigma_bar = np.sum([Ike.T @ Sigma_J @ Ike for Ike in Ikes], axis=0)
    IKPinv_sqrt = block_diag(np.eye(nx), K) @ la.inv(calP_sqrt)
    return np.sqrt(np.max(la.eigvalsh(IKPinv_sqrt.T @ Sigma_bar @ IKPinv_sqrt)))
