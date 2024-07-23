from .est_setup import EstimationSetup
from .em_runner import EMRunner
from .opt_blocks import *
from .lbfgs import lbfgs_covar
from d2pc import log_info, symmetrize, psd_inverse, log_warning


class StructuredEMRunner(EMRunner):
    def __init__(self, setup: EstimationSetup,  update_init_dist:bool = True, 
                 min_eigval: float = 1e-8, max_eigval: float = 1e8, max_abs_val: float = 1e8, ignore_Theta=False):
        """
            StructuredEMRunner class for estimating LTI system parameters subject to affine parameterizations
            - setup: EstimationSetup object
            - update_init_dist: bool whether to estimate initial distribution parameters
            - max_eigval: float maximum eigenvalue for covariance matrices
            - min_eigval: float minimum eigenvalue for covariance matrices
            - max_abs_val: float maximum absolute value for covariance matrices
            - ignore_Theta: bool whether to ignore the parameter constraints
        """
        self.sys = setup.sys
        self.update_init_dist = update_init_dist
        self.max_eigval = max_eigval
        self.min_eigval = min_eigval
        self.max_abs_val = max_abs_val
        self.ignore_Theta = ignore_Theta
        self.th0Mat = setup.th0.reshape((self.sys.nw, -1), order="F")
        self.proc_opt_blocks = create_process_opt_blocks(
            setup.Qprojectors, setup.Qtypes, setup.J, setup.th0, 
            min_eigval=self.min_eigval, max_eigval=self.max_eigval, max_abs_val=self.max_abs_val, ignore_Theta=self.ignore_Theta)
        self.Rtypes = setup.Rtypes
        self.Rprojectors = setup.Rprojectors
        # check if R needs to be updated
        self.update_R = not all([Rtype == "fixed" for Rtype in self.Rtypes])
        
    def em_m_step(self, X: np.ndarray, P: np.ndarray, Ms: np.ndarray, 
                  ys: np.ndarray, us: np.ndarray):
        """
            M-step of EM algorithm for estimating LTI system parameters
            - X: np.ndarray (T, nx) state estimates
            - P: np.ndarray (T, nx, nx) state covariances
            - Ms: np.ndarray (T-1, nx, nx) state cross-covariances
            - ys: np.ndarray (T, ny) signal of measurements
            - us: np.ndarray (T, nu) signal of inputs
            - update_init_dist: bool whether to estimate initial distribution parameters
        """
        nx, ny, nu, nw = self.sys.nx, self.sys.ny, self.sys.nu, self.sys.nw

        # calculate covariances
        z = np.concatenate([X, np.vstack([us, np.zeros((1, nu))])], axis=-1)
        Sigma = z[..., np.newaxis] @ z[:, np.newaxis]
        Sigma[:, :nx, :nx] += P
        Sigma_wo_fl = Sigma[1:-1].mean(axis=0)
        Sigma_wo_f = Sigma_wo_fl + (Sigma[-1] - Sigma_wo_fl) / (len(z)-1)
        Sigma_wo_l = Sigma_wo_fl + (Sigma[0] - Sigma_wo_fl) / (len(z)-1)

        # new sys
        new_sys = self.sys.copy()

        # ---------- init dist ---------
        if self.update_init_dist:
            mu0, P0 = X[0], symmetrize(P[0])
            # check if mu0, P0 is within the bounds
            if not self.ignore_Theta:
                # closed form solution is outside \Theta, use L-BFGS-B
                eigvalsh = la.eigvalsh(P0)
                if np.min(eigvalsh) < self.min_eigval or np.max(eigvalsh) > self.max_eigval or np.max(np.abs(mu0)) > self.max_abs_val:
                    mu0, P0 = lbfgs(self.sys.mu0, lchol(self.sys.P0), np.eye(nx), np.zeros(nx), [np.eye(nx)], ["full"],
                                    np.ones((1,1)), X[0][:, None], np.outer(X[0], X[0]),
                                    self.min_eigval, self.max_eigval, self.max_abs_val, self.ignore_Theta)
            new_sys.mu0, new_sys.P0 = mu0, P0

        # ------- R ------------
        if self.update_R:
            # new R
            Rnew = np.zeros((ny, ny))
            # statistics for R
            Psi_y = np.mean(ys[..., np.newaxis] @ X[1:, np.newaxis], axis=0)
            Phi_y = np.mean(ys[..., np.newaxis] @ ys[:, np.newaxis], axis=0)
            Sigma_y = Sigma_wo_f[:nx, :nx]
            for Rtype, Rproj in zip(self.Rtypes, self.Rprojectors):
                if Rtype == "fixed":
                    continue

                # compute statistics for current projector
                Psi_y_curr = Rproj @ Psi_y
                Phi_y_curr = Rproj @ Phi_y @ Rproj.T
                C_curr = Rproj @ new_sys.C
                R_curr = Rproj @ new_sys.R @ Rproj.T

                # use opt block to compute new _Rnew
                _Rnew = optimize_covar(
                    C_curr, Rtype, R_curr, Sigma_y, Psi_y_curr, Phi_y_curr)
                # check if _Rnew is within the bounds
                if not self.ignore_Theta:
                    # closed form solution is outside \Theta, use L-BFGS-B
                    eigvalsh = la.eigvalsh(_Rnew)
                    if np.min(eigvalsh) < self.min_eigval or np.max(eigvalsh) > self.max_eigval:
                        # try to find local minima with \Theta
                        _Rnew = lbfgs_covar(C_curr, lchol(R_curr), Rtype, Sigma_y, Psi_y_curr, Phi_y_curr,
                                            min_eigval=self.min_eigval, max_eigval=self.max_eigval, ignore_Theta=self.ignore_Theta)
                # project back
                Rnew += Rproj.T @ _Rnew @ Rproj
            # update R
            new_sys.R = Rnew

        # ----------- A, B, Q -----------
        # compute statistics
        Sigma = Sigma_wo_l
        Psi = X[1:, :, np.newaxis] @ z[:-1, np.newaxis]
        Psi[:, :, :nx] += Ms
        Psi = new_sys.E_pinv @ Psi.mean(axis=0)
        Phi = new_sys.E_pinv @ Sigma_wo_f[:nx, :nx] @ new_sys.E_pinv.T

        AB = np.hstack([new_sys.A, new_sys.B])
        E_pinv_AB = new_sys.E_pinv @ AB

        matL = np.block(
            [[np.eye(nx+nu), np.zeros((nx+nu, nw))], [-self.th0Mat, np.eye(nw)]])
        covs = np.block([[Sigma, Psi.T], [Psi, Phi]])
        covs = matL @ covs @ matL.T
        Sigma_b = covs[:nx+nu, :nx+nu]
        Psi_b = covs[nx+nu:, :nx+nu]

        Sigma_b_inv, logdetSigma_b = psd_inverse(Sigma_b, det=True)
        if logdetSigma_b > 1e7:
            log_warning("WARNING: Sigma is ill-conditioned, terminating...")
            return

        E_pinv_AB_new = np.zeros_like(E_pinv_AB)
        Q = np.zeros((nw, nw))

        for opt_block in self.proc_opt_blocks:
            projector = opt_block.projector

            # current statistics
            Q_curr = projector @ new_sys.Q @ projector.T
            E_pinv_AB_curr = projector @ E_pinv_AB
            Psi_curr = projector @ Psi
            Psi_b_curr = projector @ Psi_b
            Phi_curr = projector @ Phi @ projector.T

            # compute new parameters
            th_curr, Q_opt_curr = opt_block.optimize(
                E_pinv_AB_curr, Q_curr, Sigma, Sigma_b, Psi_curr, Psi_b_curr, Phi_curr)

            # check if Q_opt_curr is within the bounds
            if not self.ignore_Theta:
                eigvalsh = la.eigvalsh(Q_opt_curr)
                if np.min(eigvalsh) < self.min_eigval or np.max(eigvalsh) > self.max_eigval or np.max(np.abs(th_curr)) > self.max_abs_val:
                    # closed form solution is outside \Theta, use L-BFGS-B
                    J_pinv = la.pinv(opt_block.J)
                    th_curr = J_pinv @ (E_pinv_AB_curr.ravel("F") - opt_block.th0)
                    # try to find local minima with \Theta
                    th_curr, Q_opt_curr = lbfgs(th_curr, lchol(Q_curr), opt_block.J, opt_block.th0, 
                                                [np.eye(Q_opt_curr.shape[0])], [opt_block.Qtype],
                                                Sigma, Psi, Phi,
                                                self.min_eigval, self.max_eigval, self.max_abs_val, self.ignore_Theta)

            # reproject
            E_pinv_AB_curr = (
                opt_block.J @ th_curr).reshape(-1, nx+nu, order="F")
            E_pinv_AB_new += opt_block.projector.T @ E_pinv_AB_curr
            Q += opt_block.projector.T @ Q_opt_curr @ opt_block.projector

        # update A, B, Q
        AB_new = new_sys.E @ (E_pinv_AB_new + self.th0Mat)  # optimized part
        AB_new += new_sys.E_ann_pinv.T @ new_sys.E_ann.T @ AB  # fixed part
        new_sys.A, new_sys.B = AB_new[:, :nx], AB_new[:, nx:]
        new_sys.Q = Q
        # update system
        self.sys = new_sys