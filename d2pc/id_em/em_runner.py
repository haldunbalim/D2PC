from .opt_blocks import *
from .lbfgs import lbfgs_covar
from d2pc import log_info, log_warning, symmetrize, psd_inverse
from d2pc.system import LTISystem


class EMRunner:
    def __init__(self, sys: LTISystem, update_init_dist: bool = True):
        """
            EMRunner class for estimating LTI system parameters
            - setup: EstimationSetup object
        """
        self.sys = sys
        assert self.sys.nw == self.sys.nx, "nw should be equal to nx"
        assert np.allclose(self.sys.E, np.eye(self.sys.nx)), "E should be identity"
        self.update_init_dist = update_init_dist

    def em_e_step(self, ys: np.ndarray, us: np.ndarray, use_steady: bool = True):
        """
            E-step of EM algorithm for estimating LTI system parameters
            - ys: np.ndarray (T, ny) signal of measurements
            - us: np.ndarray (T, nu) signal of inputs
            - use_steady: bool whether to use steady state KF for E-step
        """
        X, P, J, lls = self.sys.kf_fwd_bwd(ys, us, use_steady=use_steady)

        # compute_Ms
        Ms = P[1:] @ np.transpose(J, (0, 2, 1))

        # expectation
        E = lls.sum()

        return X, P, Ms, E

    def em_m_step(self, X: np.ndarray, P: np.ndarray, Ms: np.ndarray,
                  ys: np.ndarray, us: np.ndarray):
        """
            M-step of EM algorithm for estimating LTI system parameters
            - X: np.ndarray (T, nx) state estimates
            - P: np.ndarray (T, nx, nx) state covariances
            - Ms: np.ndarray (T-1, nx, nx) state cross-covariances
            - ys: np.ndarray (T, ny) signal of measurements
            - us: np.ndarray (T, nu) signal of inputs
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
            new_sys.mu0, new_sys.P0 = X[0], symmetrize(P[0])

        # ------- R ------------
        Psi_y = np.mean(ys[..., np.newaxis] @ X[1:, np.newaxis], axis=0)
        Phi_y = np.mean(ys[..., np.newaxis] @ ys[:, np.newaxis], axis=0)
        Sigma_y = Sigma_wo_f[:nx, :nx]
        t = Psi_y @ new_sys.C.T
        new_sys.R = symmetrize(Phi_y + new_sys.C @ Sigma_y @ new_sys.C.T - t - t.T)

        # ----------- A, B, Q -----------
        # compute statistics
        Sigma = Sigma_wo_l
        Psi = X[1:, :, np.newaxis] @ z[:-1, np.newaxis]
        Psi[:, :, :nx] += Ms
        Psi = Psi.mean(axis=0)
        Phi = Sigma_wo_f[:nx, :nx]

        Sigma_inv, logdetSigma = psd_inverse(Sigma, det=True)
        if logdetSigma > 1e7:
            log_warning("WARNING: Sigma is ill-conditioned, terminating...")
            return
        # update A, B, Q
        AB_new = Psi @ Sigma_inv
        new_sys.A, new_sys.B = AB_new[:, :nx], AB_new[:, nx:]
        t = Psi @ AB_new.T
        new_sys.Q = symmetrize(Phi + AB_new @ Sigma.T @ AB_new.T - t - t.T)

        # update system
        self.sys = new_sys

    def em(self, ys: np.ndarray, us: np.ndarray,
           max_iter: int = 1000, rtol: float = 1e-7, verbose: int = 1,
           use_steady: bool = True, path: bool = False, stop_at_kb_interrupt: bool = True):
        """
            EM algorithm for estimating LTI system parameters
            - ys: np.ndarray (T, ny) signal of measurements
            - us: np.ndarray (T, nu) signal of inputs
            - max_iter: int maximum number of iterations
            - rtol: float relative tolerance for convergence
            - verbose: int verbosity level
            - use_steady: bool whether to use steady state KF for E-step
            - path: bool whether to return path of systems
            - stop_at_kb_interrupt: bool whether to stop at keyboard interruption or throw exception
        """

        Es = []
        systems = [self.sys.copy()]
        for i in range(max_iter):
            try:
                # expectation
                X, P, Ms, E = self.em_e_step(ys, us, use_steady=use_steady)
                Es.append(E)
                # check if expectation decreased, which should not happen
                if len(Es) > 1 and Es[-2] > Es[-1] + 2e-8:
                    log_warning("WARNING: Expectation decreased")

                # log progress
                if (verbose == 1 and i % 1000 == 0) or verbose > 1:
                    log_info(f"{i}: {E}")

                # check convergence
                if len(Es) > 1 and np.abs((Es[-1] - Es[-2]) / Es[-2]) <= rtol:
                    if verbose:
                        log_info(
                            "Terminated at iteration {} due to convergence".format(i))
                    break

                # stop if max_iter reached
                if i == max_iter-1:
                    break

                # maximization
                self.em_m_step(X=X, P=P, Ms=Ms, ys=ys, us=us)
                # save system
                if path:
                    systems.append(self.sys.copy())
            except KeyboardInterrupt:
                # stop at keyboard interruption
                if stop_at_kb_interrupt:
                    if verbose:
                        log_info(
                            "Terminated at iteration {} due to keyboard interruption".format(i))
                    break
                else:
                    raise KeyboardInterrupt()
        if verbose and i == max_iter - 1:
            log_info("Terminated at iteration {} due to max_iter".format(i))
        if path:
            return systems, Es
        else:
            return self.sys, E
