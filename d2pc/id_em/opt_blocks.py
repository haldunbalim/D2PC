import numpy as np
import scipy.linalg as la
from d2pc.utils import psd_inverse, symmetrize, lchol, inv_tril
from .lbfgs import lbfgs
from typing import List, Optional


def create_process_opt_blocks(Qprojectors:List[np.ndarray], Qtypes:List[np.ndarray], J:np.ndarray, th0:Optional[np.ndarray]=None,
                              min_eigval:float=1e-8, max_eigval:float=1e8, max_abs_val:float=1e8, ignore_Theta:bool=False):
    """
    Create optimization blocks for the process matrices
        - Qprojectors: List of projection matrices of Q blocks
        - Qtypes: List of types of Q blocks
        - J: np.ndarray (nx*(nx+nu), nth) affine parametrization matrix
        - th0: np.ndarray (nx*(nx+nu)) offset theta
        - min_eigval: float minimum eigenvalue for the covariance matrix
        - max_eigval: float maximum eigenvalue for the covariance matrix
        - max_abs_val: float maximum absolute value for theta
    """
    nw = sum([proj.shape[0] for proj in Qprojectors])
    # first obtain th0, Jsub for each Q block
    Jmat = J.T.reshape(-1, nw, J.shape[0]//nw, order="F")
    J_indvs = [proj @ Jmat for proj in Qprojectors]
    if th0 is None:
        th0_indvs = [None] * len(Qprojectors)
    else:
        th0_indvs = [(proj @ th0.reshape(nw, -1, order="F")).ravel("F")
                     for proj in Qprojectors]
    J_indvs_param = [set(np.where(J_indv)[0]) for J_indv in J_indvs]
    ignore = set()
    opt_blocks_indices = []
    # identify disjoint blocks by checking if two blocks share any parameter
    for i in range(len(Qprojectors)):
        sharing = [i]
        if i in ignore:
            continue
        for j in range(i+1, len(Qprojectors)):
            if J_indvs_param[i].isdisjoint(J_indvs_param[j]):
                continue
            else:
                sharing.append(j)
                ignore.add(j)
        opt_blocks_indices.append(sharing)
    opt_blocks = []
    J_indvs = [J_indvs.reshape(
        J.shape[-1], -1, order="F").T for J_indvs in J_indvs]
    # create optimization blocks
    for opt_blocks_idx in opt_blocks_indices:
        if len(opt_blocks_idx) == 1:
            # single block
            idx = opt_blocks_idx[0]
            J_curr = J_indvs[idx]
            J_cols = J.reshape(
                J_curr.shape[0] // Qprojectors[idx].shape[0], Qprojectors[idx].shape[0], -1)
            J_col_ranks = np.array(
                [np.linalg.matrix_rank(J_col) for J_col in J_cols])
            cf_cond = all([el in [0,  Qprojectors[idx].shape[0]] for el in J_col_ranks])
            # check if closed-form solution is possible
            if Qtypes[idx] == "full" and not cf_cond:
                # iterative solution
                opt_blocks.append(OptBlockLBFGS(Qprojectors[idx], J_indvs[idx],
                                                th0_indvs[idx], [Qtypes[idx]], [np.eye(Qprojectors[idx].shape[0])]), 
                                                min_eigval=min_eigval, max_eigval=max_eigval, 
                                                max_abs_val=max_abs_val, ignore_Theta=ignore_Theta)
            else:
                # closed-form solution
                opt_blocks.append(OptBlockCF(
                    Qtypes[idx], Qprojectors[idx], J_indvs[idx], th0_indvs[idx]))
        else:
            # mixed blocks
            Qtypes_sub = [Qtypes[idx] for idx in opt_blocks_idx]
            proj_sub = [Qprojectors[idx] for idx in opt_blocks_idx]
            proj_sub_nws = [proj.shape[0] for proj in proj_sub]
            J_sub = [J_indvs[idx] for idx in opt_blocks_idx]
            th0_sub = [th0_indvs[idx] for idx in opt_blocks_idx]
            proj_new = np.vstack(proj_sub)

            # create internal projection for mixed blocks
            cmsm = list(np.cumsum(proj_sub_nws))
            eye = np.eye(cmsm[-1])
            proj_sub = [eye[b:e] for b, e in zip([0]+cmsm[:-1], cmsm)]
            J_new = np.zeros((proj_new.shape[0] * J.shape[0] // nw, J.shape[1]))
            th0_new = np.zeros(sum([len(_th0) for _th0 in th0_sub]))
            for _p, _J, _th0 in zip(proj_sub, J_sub, th0_sub):
                _nw = _p.shape[0]
                tmp = _p.T @ _J.T.reshape(-1, _nw, _J.shape[0]// _nw, order="F")
                tmp = tmp.reshape(J.shape[-1], -1, order="F").T
                J_new += tmp
                th0_new += (_p.T @ _th0.reshape(_nw, -1, order="F")).ravel("F")
            # check if closed-form solution is possible, all Qs are fixed?
            if all([Qtype == "fixed" for Qtype in Qtypes_sub]):
                # LSTSQ solution
                opt_blocks.append(OptBlockCF("fixed", proj_new, J_new, th0_new))
            else:
                # iterative solution
                opt_blocks.append(OptBlockLBFGS(proj_new, J_new, th0_new, Qtypes_sub, proj_sub))
    return opt_blocks


class OptBlock:
    def __init__(self, projector:np.ndarray, J:np.ndarray, th0:np.ndarray):
        """
        Base class for optimization blocks
            - projector: np.ndarray (?, nw) projection matrix
            - J: np.ndarray (nx*(nx+nu), nth) affine parametrization matrix
            - th0: np.ndarray (nx*(nx+nu)) offset theta
        """
        self.projector = projector
        self.J = J
        self.th0 = th0

    def optimize(self, th, Q, Sigma, Sigma_b, Psi_curr, Psi_b_curr, Phi_curr, Phi_b_curr):
        raise NotImplementedError("Base class")


class OptBlockCF(OptBlock):
    def __init__(self, Qtype:str, projector:np.ndarray, J:np.ndarray, th0:np.ndarray):
        """
        Closed-form optimization block
            - Qtype: str type of Q block
            - projector: np.ndarray (?, nw) projection matrix
            - J: np.ndarray (nx*(nx+nu), nth) affine parametrization matrix
            - th0: np.ndarray (nx*(nx+nu)) offset theta
        """
        self.Qtype = Qtype
        subJ = J[:, ~np.all(J == 0, axis=0)]
        super().__init__(projector, subJ, th0)
        J_cols = J.reshape(
            J.shape[0] // projector.shape[0], projector.shape[0], -1)
        J_col_ranks = np.array(
            [np.linalg.matrix_rank(J_col) for J_col in J_cols])
        self.M = np.eye(J_col_ranks.shape[0])[J_col_ranks != 0]
        self.cf_cond = all([el in [0,  projector.shape[0]] for el in J_col_ranks])

    def optimize_th_b(self, Sigma_b, Psi_b, Q):
        if self.M.shape[0] == 0:
            return np.zeros((Psi_b.shape[0], Psi_b.shape[1]))
        else:
            Sigma_b = self.M @ Sigma_b @ self.M.T
            Psi_b = Psi_b @ self.M.T
            if self.cf_cond:
                # closed form sol exists
                Sigma_b_inv = psd_inverse(Sigma_b) 
                AB_next_b = Psi_b @ Sigma_b_inv
                th_next_b = AB_next_b.ravel("F")
            else:
                # scaled or fixed Q, use WLS
                Sigma_b_lchol = lchol(Sigma_b)
                Qinv_lchol = lchol(la.inv(Q))
                D = np.kron(Sigma_b_lchol.T @ self.M, Qinv_lchol) @ self.J
                d = (Qinv_lchol @ Psi_b @ inv_tril(Sigma_b_lchol).T).ravel("F")
                th_next_b = la.inv(D.T @ D) @ D.T @ d

        return th_next_b

    def optimize(self, AB_curr, Q, Sigma, Sigma_b, Psi, Psi_b, Phi):
        th_next_b = self.optimize_th_b(Sigma_b, Psi_b, Q)
        if self.Qtype == "fixed":
            Q_next = Q
        else:
            AB_next = (self.J @ th_next_b + self.th0).reshape(-1, Psi.shape[1], order="F")
            Q_next = optimize_covar(AB_next, self.Qtype, Q, Sigma, Psi, Phi)
        return th_next_b, Q_next

class OptBlockLBFGS(OptBlock):
    def __init__(self, projector:np.ndarray, J:np.ndarray, th0:np.ndarray, Qtypes:List[str], projectors:List[np.ndarray],
                  min_eigval:float=1e-8, max_eigval:float=1e8, max_abs_val=1e8, ignore_Theta:bool=False):
        """
        Optimization block with iterative solution
            - Qtypes: List of types of Q blocks
            - projectors: List of internal projection matrices of Q blocks
            - projector: np.ndarray (nw, nw) projection matrix
            - J: np.ndarray (ny, nw) affine parametrization matrix
            - th0: np.ndarray (nw, ny) initial guess for the optimization
        """
        subJ = J[:, ~np.all(J == 0, axis=0)]
        super().__init__(projector, subJ, th0)
        self.J_pinv = la.pinv(self.J)
        self.Qtypes = Qtypes
        self.projectors = projectors
        self.min_eigval = min_eigval
        self.max_eigval = max_eigval
        self.max_abs_val = max_abs_val
        self.ignore_Theta = ignore_Theta
    
    def optimize(self, AB_curr, Q, Sigma, Sigma_b, Psi, Psi_b, Phi):
        th_curr = self.J_pinv @ (AB_curr.ravel("F") - self.th0)
        th_b, Q = lbfgs(th_curr, lchol(Q), self.J, self.th0, self.projectors, self.Qtypes, Sigma, Psi, Phi,
                         self.min_eigval, self.max_eigval, self.max_abs_val, self.ignore_Theta)
        return th_b, Q
    
def optimize_covar(regressor:np.ndarray, typ:str, cov_curr:np.ndarray, Sigma:np.ndarray, Psi:np.ndarray, Phi:np.ndarray):
    """
    Optimize the covariance matrix
        - regressor: np.ndarray regressor matrix
        - typ: str type of the covariance matrix to be returned
        - cov_curr: np.ndarray (nx+nu, nx+nu) current covariance matrix
        - Sigma: np.ndarray input covariance matrix
        - Psi: np.ndarray cross-covariance matrix
        - Phi: np.ndarray output covariance matrix
    """
    t = Psi @ regressor.T
    cov_next = symmetrize(Phi + regressor @ Sigma @ regressor.T - t - t.T)
    if typ == "scaled":
        cov_next = compute_scaled_covar(cov_next, cov_curr)
    return cov_next

def compute_scaled_covar(cov, base_cov):
    return np.trace(psd_inverse(base_cov) @ cov) / base_cov.shape[0] * base_cov
