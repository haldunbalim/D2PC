
import numpy as np
from functools import partial
import scipy.linalg as la
from scipy.optimize import minimize, LinearConstraint
from d2pc.utils import inv_tril, log_warning
from typing import List


def lbfgs(th_curr: np.ndarray, cov_chol: np.ndarray, J: np.ndarray, th0: np.ndarray,
          projectors: List[np.ndarray], Qtypes: List[str],
          Sigma: np.ndarray, Psi: np.ndarray, Phi: np.ndarray,
          min_eigval: float = 1e-8, max_eigval: float = 1e8,
          max_abs_val: float = 1e8, ignore_Theta: bool = False):
    """
        A function to maximize conditional likelihood using L-BFGS-B.
        - th_curr: The initial value of the parameter vector.
        - cov_chol: The initial value of the Cholesky factor of the noise covariance matrix.
        - J: The Jacobian of the system dynamics.
        - th0: The offset of the system dynamics.
        - projectors: The projectors for the noise covariance matrix.
        - Qtypes: The types of the noise covariance matrix.
        - Sigma: The state-input covariance matrix.
        - Psi: The state-input-output covariance matrix.
        - Phi: The output covariance matrix.
        - min_eigval: The minimum eigenvalue of the noise covariance matrix.
        - max_eigval: The maximum eigenvalue of the noise covariance matrix.
        - max_abs_val: The maximum absolute value of the parameter vector.
        - ignore_Theta: A flag to ignore the parameter vector.
    """
    _obj_w_jac = partial(obj_w_jac, J=J, th0=th0, projectors=projectors, Qtypes=Qtypes,
                         Sigma=Sigma, Psi=Psi, Phi=Phi)

    bounds = None
    if not ignore_Theta:
        bounds = [(-max_abs_val, max_abs_val) for _ in range(J.shape[1])]
        nw = cov_chol.shape[0]
        idx = set(np.diag(np.arange(nw**2).reshape(nw, nw, order="F")))
        bounds += [(np.sqrt(min_eigval), np.sqrt(max_eigval))
                   if i in idx else (None, None) for i in range(nw**2)]

    res = minimize(_obj_w_jac, x0=np.concatenate([th_curr, cov_chol.ravel("F")]),
                   jac=True, method="L-BFGS-B",
                   tol=1e-7, options={"maxiter": 5000, "disp": False}, bounds=bounds)
    
    cov_chol_new = res.x[J.shape[1]:].reshape(Phi.shape[0], -1, order="F")
    cov_new = cov_chol_new @ cov_chol_new.T
    # Sometimes numerical issues occur here yielding cov to have min eigval below threshold
    # Although the optimization is successful
    # In such cases, we return the initial values
    if res.success and np.all(la.eigvalsh(cov_new) >= min_eigval):
        cov = cov_new
        th_curr = res.x[:J.shape[1]]
    else:
        cov = cov_chol @ cov_chol.T

    return th_curr, cov


def obj_w_jac(param_vec: np.ndarray, J: np.ndarray, th0: np.ndarray,
              projectors: List[np.ndarray], Qtypes: List[str],
              Sigma: np.ndarray, Psi: np.ndarray, Phi: np.ndarray):
    """
        A function to compute the objective function and its gradient.
        - param_vec: The parameter vector.
        - J: The Jacobian of the system dynamics.
        - th0: The offset of the system dynamics.
        - projectors: The projectors for the noise covariance matrix.
        - Qtypes: The types of the noise covariance matrix.
        - Sigma: The state-input covariance matrix.
        - Psi: The state-input-output covariance matrix.
        - Phi: The output covariance matrix.
    """
    # recover the parameters
    th = param_vec[:J.shape[1]]
    AB = J @ th + th0
    AB = AB.reshape(-1, Psi.shape[1], order="F")
    Q_chol = param_vec[J.shape[1]:].reshape(Phi.shape[0], -1, order="F")
    # compute the inverse of the noise covariance matrix
    Q_chol_inv = inv_tril(Q_chol)
    Qinv = Q_chol_inv.T @ Q_chol_inv
    # compute the objective function and its gradient
    t = Psi @ AB.T
    tr = Phi - t - t.T + AB @ Sigma @ AB.T
    obj = np.trace(Qinv @ tr) - 2 * np.log(np.prod(np.diag(Q_chol_inv)))
    d_obj_d_AB = (2 * Qinv @ (AB @ Sigma - Psi))
    d_obj_d_Q_lchol = (
        2 * np.eye(Qinv.shape[0]) - Qinv @ (tr + tr.T)) @ Q_chol_inv.T

    # project the gradient
    nw = Phi.shape[0]
    d_obj_d_Q_lchol[np.triu_indices(nw, 1)] = 0
    d_obj_d_Q_lchol_new = np.zeros_like(d_obj_d_Q_lchol)
    for proj, Qtype in zip(projectors, Qtypes):
        d_obj_d_Q_lchol_sub = proj @ d_obj_d_Q_lchol @ proj.T
        if Qtype == "scaled":
            Q_chol_sub = proj @ Q_chol @ proj.T
            d_obj_d_Q_lchol_sub = np.trace(
                d_obj_d_Q_lchol_sub @ Q_chol_sub) * Q_chol_sub
        elif Qtype == "fixed":
            d_obj_d_Q_lchol_sub = np.zeros_like(d_obj_d_Q_lchol_sub)
        d_obj_d_Q_lchol_new += proj.T @ d_obj_d_Q_lchol_sub @ proj
    # concatenate the gradients
    d_obj_d_param = np.concatenate(
        [J.T @ d_obj_d_AB.ravel("F"), d_obj_d_Q_lchol_new.ravel("F")])
    return obj, d_obj_d_param


def lbfgs_covar(reg: np.ndarray, cov_chol: np.ndarray,
                cov_type: str, Sigma: np.ndarray, Psi: np.ndarray, Phi: np.ndarray,
                min_eigval: float = 1e-8, max_eigval: float = 1e8, ignore_Theta: bool = False):
    """
        A function to maximize conditional likelihood using L-BFGS-B only for the noise covariance matrix.
        - reg: The value for the regressor
        - cov_chol: The initial value of the Cholesky factor of the noise covariance matrix.
        - cov_type: The type of the noise covariance matrix.
        - Sigma: The state-input covariance matrix.
        - Psi: The state-input-output covariance matrix.
        - Phi: The output covariance matrix.
        - min_eigval: The minimum eigenvalue of the noise covariance matrix.
        - max_eigval: The maximum eigenvalue of the noise covariance matrix.
        - ignore_Theta: A flag to ignore the parameter vector.
    """
    t = Psi @ reg.T
    tr = Phi - t - t.T + reg @ Sigma @ reg.T
    nw = tr.shape[0]

    _obj_w_jac_covar = partial(obj_w_jac_covar, tr=tr, cov_type=cov_type)

    bounds = None
    if not ignore_Theta:
        idx = set(np.diag(np.arange(nw**2).reshape(nw, nw, order="F")))
        bounds = [(np.sqrt(min_eigval), np.sqrt(max_eigval))
                  if i in idx else (None, None) for i in range(nw**2)]
    res = minimize(_obj_w_jac_covar, x0=cov_chol.ravel("F"), jac=True, method="L-BFGS-B",
                   tol=1e-8, options={"maxiter": 5000, "disp": False}, bounds=bounds)
    if res.success:
        cov_chol = res.x.reshape(Phi.shape[0], -1, order="F")
    else:
        log_warning("L-BFGS failed to converge. Returning the initial values.")
    return cov_chol @ cov_chol.T


def obj_w_jac_covar(param_vec: np.ndarray, tr: np.ndarray, cov_type: str):
    """
        A function to compute the objective function and its gradient for the noise covariance matrix.
        - param_vec: The Cholesky factor of the noise covariance matrix.
        - tr: The residual matrix.
        - cov_type: The type of the noise covariance matrix.
    """
    # recover the parameters
    nw = tr.shape[0]
    cov_chol = param_vec.reshape(nw, -1, order="F")
    # compute the inverse of the noise covariance matrix
    cov_chol_inv = inv_tril(cov_chol)
    cov_inv = cov_chol_inv.T @ cov_chol_inv

    # compute the objective function and its gradient
    obj = np.trace(cov_inv @ tr) - 2 * \
        np.log(np.prod(np.diag(cov_chol_inv)))
    d_obj_d_cov_chol = (
        2 * np.eye(cov_chol.shape[0]) - cov_inv @ (tr + tr.T)) @ cov_chol_inv.T
    # project the gradient
    d_obj_d_cov_chol[np.triu_indices(nw, 1)] = 0
    if cov_type == "scaled":
        d_obj_d_cov_chol = np.trace(d_obj_d_cov_chol @ cov_chol) * cov_chol
    return obj, d_obj_d_cov_chol.ravel("F")
