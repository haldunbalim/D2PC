import numpy as np
import scipy.linalg as la
import control as ct
from d2pc.utils import *
from d2pc.system import LTISystem
from scipy.stats import chi2
from .dynof_ctrl import DynOFController
from typing import Union, List
from .utils import constraint_lpv_fixed_ctrlr
from typing import Optional


def synth_dynof_ctrl(sys: LTISystem,
                     costQ: np.ndarray, costR: np.ndarray) -> DynOFController:
    """
    Synthesize a dynamic output-feedback controller for the given system.
    Uses LQG solution to compute the controller.
        - costQ: np.ndarray (nx, nx) - state cost
        - costR: np.ndarray (nu, nu) - input cost
    """
    K = -ct.dlqr(sys.A, sys.B, costQ, costR)[0]
    L = ct.dlqe(sys.A, sys.E, sys.C, sys.Q, sys.R)[0]
    Ac = sys.A + sys.B @ K - L @ sys.C

    return DynOFController(sys, Ac, K, L)


def an_dynof_ctrl(sys: LTISystem, dynof: DynOFController,
                  costQ: np.ndarray, costR: np.ndarray) -> float:
    """
    Analyze a dynamic output-feedback controller for the given system.
    Returns the H2 norm of the closed-loop system.
        - sys: LTISystem - system
        - dynof: DynOFController - dynamic output-feedback controller
        - costQ: np.ndarray (nx, nx) - state cost
        - costR: np.ndarray (nu, nu) - input cost
    """
    BdBdT = block_diag(sys.E@sys.Q@sys.E.T, dynof.L @ sys.R @ dynof.L.T)
    CepsT_Ceps = block_diag(costQ, dynof.K.T @ costR @ dynof.K)
    _calX = cp.Variable((2*sys.nx, 2*sys.nx), PSD=True)
    obj = cp.Minimize(cp.trace(_calX @ CepsT_Ceps))

    IAT = np.vstack([np.eye(2*sys.nx), dynof.calA.T])
    prob = cp.Problem(
        obj, [IAT.T @ block_diag_cvxpy(BdBdT - _calX, _calX) @ IAT << 0])
    return np.sqrt(prob.solve()), _calX.value


def synth_robust_dynof_ctrl(sys: LTISystem, J: np.ndarray, hess_th: np.ndarray, delta: float,
                            costQ: np.ndarray, costR: np.ndarray,
                            Lambda0_or_Ctrl0: Optional[np.ndarray |
                                                       DynOFController] = None,
                            max_iter: int = 500, rtol: float = 1e-6, atol: float = 5e-8,
                            path=False, D: Optional[np.ndarray] = None) \
        -> Optional[Union[float, DynOFController, np.ndarray, np.ndarray]]:
    """
        Alternating robust H2 analysis and synthesis (D-K iteration).
        - sys: LTISystem - system
        - J: np.ndarray (nw*(nx+nu), nth) - affine parametrization matrix
        - hess_th: np.ndarray (nth, nth) - Hessian of the uncertainty
        - delta: float - probability level for parametric uncertainty
        - costQ: np.ndarray (nx, nx) - state cost
        - costR: np.ndarray (nu, nu) - input cost
        - Lambda0_or_Ctrl0: np.ndarray | DynOFController - initial Lambda0 or controller
                            if None the non-robust controller is used as initialization
        - max_iter: int - maximum number of iterations
        - rtol: float - relative tolerance for stopping condition
        - atol: float - absolute tolerance for stopping condition
        - D: np.ndarray (nth, nth) - shape matrix for the approximate uncertainty set
    """
    prob_synth = construct_robust_h2_dynof_synth(
        sys, J, hess_th, delta, costQ, costR, D=D)
    prob_an = construct_robust_h2_dynof_an(
        sys, J, hess_th, delta, costQ, costR, D=D)

    rets = []
    if Lambda0_or_Ctrl0 is None:
        ctrlr = synth_dynof_ctrl(sys, costQ, costR)
    elif isinstance(Lambda0_or_Ctrl0, DynOFController):
        ctrlr = Lambda0_or_Ctrl0
    else:
        ctrlr = prob_synth(Lambda0_or_Ctrl0)
        if ctrlr is None:
            raise Exception("Problem is infeasible for initial Lambda")

    for _ in range(max_iter):
        # stop condition
        if len(rets) > 1:
            op2, op1 = rets[-2][0], rets[-1][0]
            obj_diff = op2 - op1
            rcond = obj_diff / op2 < rtol
            acond = obj_diff < atol
            if rcond or acond:
                break

        # robust analysis
        if (ret := prob_an(ctrlr)) is not None:
            obj, calX, Lambda = ret
            rets.append((obj, ctrlr, Lambda, calX))
        else:
            break

        # robust synthesis
        if (ret := prob_synth(Lambda)) is not None:
            ctrlr = ret
        else:
            break

    if len(rets) == 0:
        raise Exception("Failed to synthesize a robust controller.")
    if path:
        return rets
    else:
        return rets[-1]


def construct_robust_h2_dynof_synth(sys: LTISystem, J: np.ndarray,
                                    hess_th: np.ndarray, delta: float,
                                    costQ: np.ndarray, costR: np.ndarray,
                                    D: Optional[np.ndarray] = None):
    """
        Construct the robust H2 dynamic output feedback synthesis problem.
        - sys: LTISystem - system
        - J: np.ndarray (nw*(nx+nu), nth) - affine parametrization matrix
        - hess_th: np.ndarray (nth, nth) - Hessian of the uncertainty
        - delta: float - confidence level
        - costQ: np.ndarray (nx, nx) - state cost
        - costR: np.ndarray (nu, nu) - input cost
        - D: np.ndarray (nth, nth) - shape matrix for the approximate uncertainty set
                                     if provided the approximate uncertainty set is used
    """
    assert delta > 0 and delta < 1, "delta must be in (0, 1)"
    cdelta = chi2.ppf(delta, hess_th.shape[0])
    nx, nu, ny, nw = sys.nx, sys.nu, sys.ny, sys.nw

    # vars
    calW = cp.Variable((nx+nu, nx+nu), PSD=True)
    X = cp.Variable((nx, nx), PSD=True)
    Y = cp.Variable((nx, nx), PSD=True)
    M = cp.Variable((nu, nx))
    F = cp.Variable((nx, ny))
    S = cp.Variable((nx, nx))

    Ce = np.vstack([la.sqrtm(costQ), np.zeros((nu, nx))])
    De = np.vstack([np.zeros((nx, nu)), la.sqrtm(costR)])
    CqcalXT = cp.bmat([[X, np.eye(nx)], [M, np.zeros((nu, nx))]])
    if D is None:
        Lambda_cp = cp.Parameter((nw, nw), PSD=True)
        P_up = dpp_kron(Lambda_cp, hess_th / cdelta)
        P_low = Lambda_cp
        CqcalXT = compute_JDelta(J, nw) @ CqcalXT
    else:
        Lambda_cp = cp.Parameter(pos=True)
        P_up = D / cdelta * Lambda_cp
        P_low = np.eye(nw) * Lambda_cp

    # matrices
    TcalXT = cp.bmat([[X, np.eye(nx)], [np.eye(nx), Y]])
    TAcalXT = cp.bmat([[sys.A@X+sys.B@M, sys.A], [S, Y@sys.A+F@sys.C]])
    Ceps_calXT = np.hstack(
        [Ce, De]) @ cp.bmat([[X, np.eye(nx)], [M, np.zeros((nu, nx))]])
    Q_lchol, R_lchol = lchol(sys.Q), lchol(sys.R)
    TBd = cp.bmat([[sys.E @ Q_lchol, np.zeros((nx, ny))],
                   [Y@sys.E @ Q_lchol, F @ R_lchol]])
    TBp = cp.vstack([sys.E, Y @ sys.E])

    # constraints
    nwnth = CqcalXT.shape[0]
    r1 = [-TcalXT, np.zeros((2*nx, nwnth)), TAcalXT, TBd, TBp@P_low]
    r2 = [r1[1].T, -P_up, CqcalXT,
          np.zeros((nwnth, nw+ny)), np.zeros((nwnth, nw))]
    r3 = [r1[2].T, r2[2].T, -TcalXT,
          np.zeros((2*nx, nw+ny)), np.zeros((2*nx, nw))]
    r4 = [r1[3].T, r2[3].T, r3[3].T, -np.eye(nw+ny), np.zeros((nw+ny, nw))]
    r5 = [r1[4].T, r2[4].T, r3[4].T, r4[4].T, -P_low]
    stab_cond = cp.bmat([r1, r2, r3, r4, r5])

    perf_cond = cp.bmat([[calW, Ceps_calXT], [Ceps_calXT.T, TcalXT]])
    constraints = [stab_cond << 0, perf_cond >> 0]
    prob = cp.Problem(cp.Minimize(cp.trace(calW)), constraints)

    def solve(Lambda: np.ndarray) -> DynOFController:
        Lambda_cp.value = Lambda
        try:
            prob.solve()
        except cp.SolverError:
            return None
        if prob.status == cp.OPTIMAL:
            return synth_params_to_dynof(sys, X.value, Y.value, M.value, F.value, S.value)
    return solve


def construct_robust_h2_dynof_an(sys: LTISystem, J: np.ndarray, hess_th: np.ndarray,
                                 delta: float, costQ: np.ndarray, costR: np.ndarray,
                                 D: Optional[np.ndarray] = None):
    """
        Construct the robust H2 dynamic output feedback analysis problem.
        - sys: LTISystem - system
        - J: np.ndarray (nw*(nx+nu), nth) - affine parametrization matrix
        - _th: np.ndarray (nth, nth) - Hessian of the uncertainty
        - delta: float - confidence level
        - costQ: np.ndarray (nx, nx) - state cost
        - costR: np.ndarray (nu, nu) - input cost
        - D: np.ndarray (nth, nth) - shape matrix for the approximate uncertainty set
                                     if provided the approximate uncertainty set is used
    """
    nx = sys.nx
    # seperately defined to make problem dpp
    KTRcK = cp.Parameter((nx, nx), PSD=True)
    LRLT = cp.Parameter((nx, nx), PSD=True)
    # vars
    calX = cp.Variable((2*nx, 2*nx), PSD=True)
    # matrices
    CepsT_Ceps = block_diag_cvxpy(costQ, KTRcK)
    BdBdT = block_diag_cvxpy(sys.E @ sys.Q @ sys.E.T, LRLT)
    covar_pre = calX
    covar_post = calX - BdBdT
    # constraints
    stab_cond, Lambda, (Ac, K, L) = constraint_lpv_fixed_ctrlr(
        sys, covar_pre, covar_post, J, hess_th, delta, D=D)
    constraints = [stab_cond << 0]
    prob = cp.Problem(cp.Minimize(cp.trace(calX@CepsT_Ceps)), constraints)

    def solve(dynof: DynOFController):
        Ac.value, K.value, L.value = dynof.Ac, dynof.K, dynof.L
        KTRcK.value = dynof.K.T @ costR @ dynof.K
        LRLT.value = dynof.L @ sys.R @ dynof.L.T
        try:
            obj = prob.solve()
        except cp.SolverError:
            return None
        if prob.status == cp.OPTIMAL:
            return np.sqrt(obj), calX.value, Lambda.value
    return solve


def synth_params_to_dynof(sys: LTISystem, X: np.ndarray, Y: np.ndarray, M: np.ndarray,
                          F: np.ndarray, S: np.ndarray, V: Optional[np.ndarray] = None)\
        -> DynOFController:
    """
        Recover a dynamic output-feedback controller from the given synthesis parameterization.
        - sys: LTISystem - system
        - X: np.ndarray (nx, nx)
        - Y: np.ndarray (nx, nx)
        - M: np.ndarray (nu, nx)
        - F: np.ndarray (nx, ny)
        - S: np.ndarray (nx, nx)
        - V: np.ndarray (nx, nx)
    """

    # V can be picked as arbitrary full rank matrix
    # if not provided, we pick it as identity
    V = V if V is not None else np.eye(sys.nx)
    assert np.linalg.matrix_rank(V) == V.shape[0], "V must be full rank"

    # recover controller
    Vinv = la.inv(V)
    U = Vinv - Vinv @ Y @ X
    Uinv = la.inv(U)

    K = M @ Uinv
    L = Vinv @ F
    Ac = Vinv @ (S - Y @ sys.A @ X - F @ sys.C @ X - Y @ sys.B @ M) @ Uinv

    return DynOFController(sys, Ac, K, L)
