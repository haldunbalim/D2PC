import numpy as np
import scipy.linalg as la
from d2pc.utils import *
from scipy.linalg import solve_discrete_are
from copy import deepcopy
from typing import Optional


class LTISystem:
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, E: Optional[np.ndarray] = None,
                 Q: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None,
                 mu0: Optional[np.ndarray] = None, P0: Optional[np.ndarray] = None):
        assert A.shape[0] == A.shape[1], "A must be square"
        self._A = A
        assert B.shape[0] == A.shape[0], "B must have same number of rows as A"
        self._B = B
        assert C.shape[1] == A.shape[0], "C must have same number of columns as A"
        self._C = C

        if E is None:
            self._E = np.eye(self.nx)
            self._E_pinv = np.eye(self.nx)
            self._E_ann = np.zeros((self.nx, self.nx))
            self._E_ann_pinv = np.eye(self.nx)
        else:
            assert E.shape[0] == self.nx, "E must have same number of rows as A"
            assert E.shape[1] <= E.shape[0], "E must be tall, or square"
            assert np.linalg.matrix_rank(
                E) == E.shape[1], "E must have full column rank"
            self._E = E
            self._E_pinv = la.pinv(E)
            if E.shape[1] == E.shape[0]:
                self._E_ann = np.zeros((E.shape[1], E.shape[0]))
                self._E_ann_pinv = np.eye(E.shape[0])
            else:
                self._E_ann = la.null_space(E.T)
                self._E_ann_pinv = la.pinv(self._E_ann)

        if Q is None:
            self._Q = np.eye(self.nw)
        else:
            self._Q = Q if Q is not None else np.eye(self.nw)
            assert_dim_and_pd(self.Q, self.nw, "Q")

        self._R = R if R is not None else np.eye(self.ny)
        assert_dim_and_pd(self.R, self.ny, "R")

        # initial condition
        self._mu0 = mu0 if mu0 is not None else np.zeros(self.nx)
        assert self.mu0.shape == (self.nx,), "mu0 must be of shape (nx,)"
        self._P0 = P0 if P0 is not None else np.eye(self.nx) * 1e-2
        assert_dim_and_pd(self.P0, self.nx, "P0", allow_sd=True)

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, A):
        assert A.shape[0] == A.shape[1], "A must be square"
        assert self.B.shape[0] == A.shape[0], "B must have same number of rows as A"
        assert self.C.shape[1] == A.shape[0], "C must have same number of columns as A"
        self._A = A

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, B):
        assert B.shape[0] == self.A.shape[0], "B must have same number of rows as A"
        self._B = B

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, C):
        assert C.shape[1] == self.A.shape[0], "C must have same number of columns as A"
        self._C = C

    @property
    def E(self):
        return self._E

    @E.setter
    def E(self, E):
        raise NotImplementedError("E cannot be changed")

    @property
    def E_pinv(self):
        return self._E_pinv

    @E_pinv.setter
    def E_pinv(self, E_pinv):
        raise NotImplementedError("E_pinv cannot be changed")

    @property
    def E_ann(self):
        return self._E_ann

    @E_ann.setter
    def E_ann(self, E_ann):
        raise NotImplementedError("E_ann cannot be changed")

    @property
    def E_ann_pinv(self):
        return self._E_ann_pinv

    @E_ann_pinv.setter
    def E_ann_pinv(self, E_ann_pinv):
        raise NotImplementedError("E_ann_pinv cannot be changed")

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, Q):
        assert Q.shape == (self.nw, self.nw), "Q must be of shape (nw, nw)"
        assert la.ishermitian(Q), "Q must be Hermitian"
        assert np.all(la.eigvalsh(Q) > 0), "Q must be positive definite"
        self._Q = Q

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, R):
        assert R.shape == (self.ny, self.ny), "R must be of shape (ny, ny)"
        assert la.ishermitian(R), "R must be Hermitian"
        assert np.all(la.eigvalsh(R) > 0), "R must be positive definite"
        self._R = R

    @property
    def mu0(self):
        return self._mu0

    @mu0.setter
    def mu0(self, mu0):
        assert mu0.shape == (self.nx,), "mu0 must be of shape (nx,)"
        self._mu0 = mu0

    @property
    def P0(self):
        return self._P0

    @P0.setter
    def P0(self, P0):
        assert P0.shape == (self.nx, self.nx), "P0 must be of shape (nx, nx)"
        assert la.ishermitian(P0), "P0 must be Hermitian"
        self._P0 = P0

    @property
    def nx(self):
        return self.A.shape[0]

    @property
    def ny(self):
        return self.C.shape[0]

    @property
    def nu(self):
        return self.B.shape[1]

    @property
    def nw(self):
        return self.E.shape[1]
    
    @property
    def is_controllable(self):
        W = [self.B]
        for _ in range(self.A.shape[0]-1):
            W.append(self.A@W[-1])
        return np.linalg.matrix_rank(np.hstack(W)) == self.A.shape[0]

    @property
    def is_observable(self):
        W = [self.C]
        for _ in range(self.A.shape[0]-1):
            W.append(W[-1]@self.A)
        return np.linalg.matrix_rank(np.vstack(W)) == self.A.shape[0]

    def new_AB_from_th(self, J, th, th0=None):
        E_pinv_AB = J@th
        if th0 is not None:
            E_pinv_AB += th0
        E_pinv_AB = E_pinv_AB.reshape(self.nw, -1, order="F")
        AB = np.hstack([self.A, self.B])
        AB = self.E @ E_pinv_AB + self.E_ann_pinv.T @ self.E_ann.T @ AB
        return AB[:, :self.nx], AB[:, self.nx:]

    def simulate(self, ts, controller, x0=None, ws=None, vs=None, return_noises=False):
        xs = [np.random.multivariate_normal(
            self.mu0, self.P0) if x0 is None else x0]
        ys = []
        us = []
        if ws is None:
            ws = np.random.multivariate_normal(
                np.zeros(self.nw), self.Q, size=ts)
        assert ws.shape == (ts, self.nw), "ws must be of shape (ts, nw)"

        if vs is None:
            vs = np.random.multivariate_normal(
                np.zeros(self.ny), self.R, size=ts)
        assert vs.shape == (ts, self.ny), "vs must be of shape (ts, ny)"

        for i in range(ts):
            x = xs[-1]
            u = controller()
            us.append(u)
            xs.append(self.A @ x + self.B @ u + self.E @ ws[i])
            ys.append(self.C @ xs[-1] + vs[i])

        xs = np.array(xs).astype(np.float64)
        us = np.array(us).astype(np.float64)
        ys = np.array(ys).astype(np.float64)
        ret = xs, ys, us
        if return_noises:
            ret += (ws, vs)
        return ret

    def copy(self):
        return deepcopy(self)
    
    def transform(self, T):
        Tinv = la.inv(T)
        A = T @ self.A @ Tinv
        B = T @ self.B
        C = self.C @ Tinv
        E = T @ self.E
        mu0 = T @ self.mu0
        P0 = symmetrize(T @ self.P0 @ T.T)
        return LTISystem(A, B, C, E, self.Q, self.R, mu0, P0)
    
    def __eq__(self, other):
        if not isinstance(other, LTISystem):
            return False
        if self.nx != other.nx or self.ny != other.ny or self.nu != other.nu or self.nw != other.nw:
            return False
        if np.allclose(self.A, other.A) and np.allclose(self.B, other.B) and np.allclose(self.C, other.C)\
              and np.allclose(self.E, other.E) and np.allclose(self.Q, other.Q) and np.allclose(self.R, other.R)\
                  and np.allclose(self.mu0, other.mu0) and np.allclose(self.P0, other.P0, rtol=1e-4):
            return True
        return False
    
    # ----------------- Kalman Filter -----------------
    def kf_update(self, y, x_prior, P_prior):
        C, R = self.C, self.R

        PCT = P_prior @ C.T
        S = C @ PCT + R
        Sinv, logdetSt = psd_inverse(S, det=True)
        ny, nx = C.shape

        K = PCT @ Sinv
        e = (y - C @ x_prior)
        x_post = x_prior + K @ e
        P_post = (np.eye(nx) - K @ C) @ P_prior
        ll = e.T @ Sinv @ e + logdetSt + ny * np.log(2 * np.pi)
        return x_post, symmetrize(P_post), -0.5*ll, e, S, Sinv

    def kf_predict(self, x_post, P_post, u):
        A, B, E, Q = self.A, self.B, self.E, self.Q
        P_prior = A @ P_post @ A.T + E @ Q @ E.T
        x_prior = A @ x_post + B @ u
        return x_prior, symmetrize(P_prior)

    def kf_fwd(self, ys, us, return_aux=False, use_steady=False):
        nx, ny = self.nx, self.ny

        x_priors = np.zeros((len(ys), nx))
        P_priors = np.zeros((len(ys), nx, nx))
        x_posts = np.zeros((len(ys)+1, nx))
        x_posts[0] = self.mu0
        P_posts = np.zeros((len(ys)+1, nx, nx))
        P_posts[0] = self.P0
        lls = np.zeros(len(ys))
        es = np.zeros((len(ys), ny))
        Ss = np.zeros((len(ys), ny, ny))
        Sinvs = np.zeros((len(ys), ny, ny))
        if use_steady:
            try:
                P_prior_steady = self.get_steady_prior_covar()
            except:
                use_steady = False

        for t, (y, u) in enumerate(zip(ys, us)):
            # predict
            x_priors[t], P_priors[t] = self.kf_predict(
                x_posts[t], P_posts[t], u)

            # update
            x_posts[t+1], P_posts[t+1], lls[t], es[t], Ss[t], Sinvs[t] = self.kf_update(
                y, x_priors[t],  P_priors[t])

            # save
            if use_steady and np.allclose(P_priors[t], P_prior_steady, atol=1e-8, rtol=0):
                (x_posts[t+2:], P_posts[t+2:], x_priors[t+1:], P_priors[t+1:],
                 lls[t+1:], es[t+1:], Ss[t+1:], Sinvs[t+1:]) = self.steady_filter(ys[t+1:], us[t+1:],
                                                                                  x_posts[t+1], P_prior_steady)
                break

        ret = x_posts, P_posts, x_priors, P_priors, lls
        if return_aux:
            ret += (es, Ss, Sinvs)
        return ret

    def get_kf_fwd_steady_idx(self, max=100):
        P_prior_steady = self.get_steady_prior_covar()
        A, C, E, Q, R = self.A, self.C, self.E, self.Q, self.R
        P_post = self.P0
        nx = self.nx
        for t in range(max):
            P_prior = A @ P_post @ A.T + E @ Q @ E.T
            if np.allclose(P_prior, P_prior_steady, atol=1e-8, rtol=0):
                return t

            PCT = P_prior @ C.T
            Sinv = psd_inverse(C @ PCT + R)
            K = PCT @ Sinv
            P_post = (np.eye(nx) - K @ C) @ P_prior
        return -1

    def rts_smooth_iter(self, P_next, x_next, x_post, u, P_post, P_prior):
        P_prior_inv = psd_inverse(P_prior)
        J = P_post @ self.A.T @ P_prior_inv
        x = x_post + J @ (x_next - self.A @ x_post - self.B @ u)
        P = P_post + J @ (P_next - P_prior) @ J.T
        return J, x, P

    def rts_smooth_iter_steady(self, P_next, x_next, x_post, u, P_post, P_prior, J):
        x = x_post + J @ (x_next - self.A@x_post - self.B @ u)
        P = P_post + J @ (P_next - P_prior) @ J.T
        return x, P

    def kf_bwd(sys, us, x_posts, P_posts, P_priors, use_steady=False):
        n, nx, _ = P_priors.shape

        # smoother gain
        J = np.zeros((n, nx, nx))
        X, P = np.zeros_like(x_posts), np.zeros_like(P_posts)

        X[-1] = x_posts[-1]
        P[-1] = P_posts[-1]
        for t in reversed(range(n)):
            if use_steady and t < n-2 and \
                np.allclose(P_posts[t+1], P_posts[t], rtol=0, atol=1e-8) and \
                    np.allclose(P_priors[t+1], P_priors[t+2], rtol=0, atol=1e-8):
                J[t] = J[t+1]
                X[t], P[t] = sys.rts_smooth_iter_steady(
                    P[t+1], X[t+1], x_posts[t], us[t], P_posts[t], P_priors[t+1], J[t])
            else:
                J[t], X[t], P[t] = sys.rts_smooth_iter(
                    P[t+1], X[t+1], x_posts[t], us[t], P_posts[t], P_priors[t])

        return (X, P, J)

    def get_steady_prior_covar(self):
        return symmetrize(solve_discrete_are(self.A.T, self.C.T, self.E@self.Q@self.E.T, self.R))
    
    def get_steady_post_covar(self):
        P_prior_steady = self.get_steady_prior_covar()
        PCT = P_prior_steady @ self.C.T
        S_steady = symmetrize(self.C @ PCT + self.R)
        S_inv_steady = psd_inverse(S_steady)
        S_inv_steady = symmetrize(S_inv_steady)
        K_steady = PCT @ S_inv_steady
        P_post_steady = (np.eye(self.nx) - K_steady @ self.C) @ P_prior_steady
        return symmetrize(P_post_steady)

    def steady_filter(self, ys, us, x_post, P_prior_steady):
        A, B, C, R = self.A, self.B, self.C, self.R
        PCT = P_prior_steady @ C.T
        S_steady = symmetrize(C @ PCT + R)
        S_inv_steady, logdetSt = psd_inverse(S_steady, det=True)
        S_inv_steady = symmetrize(S_inv_steady)
        K_steady = PCT @ S_inv_steady
        P_post_steady = symmetrize(
            (np.eye(self.nx) - K_steady @ C) @ P_prior_steady)

        x_priors = np.zeros((len(ys), self.nx))
        x_posts = np.zeros((len(ys)+1, self.nx))
        x_posts[0] = x_post
        es = np.zeros((len(ys), self.ny))
        lls = np.zeros(len(ys))
        ny = self.ny
        for t, (y, u) in enumerate(zip(ys, us)):
            # predict
            x_priors[t] = A @ x_posts[t] + B @ u

            # update
            es[t] = y - C @ x_priors[t]
            x_posts[t+1] = x_priors[t] + K_steady @ es[t]

        lls = np.squeeze(es[:, None] @ S_inv_steady @ es[..., None])
        lls = -.5 * (lls + logdetSt + ny*np.log(2*np.pi))
        return x_posts[1:], P_post_steady, x_priors, P_prior_steady, lls, es, S_steady, S_inv_steady

    def kf_fwd_bwd(self, ys, us, use_steady=False):
        x_posts, P_posts, _, P_priors, lls = self.kf_fwd(
            ys, us, use_steady=use_steady)
        X, P, J = self.kf_bwd(us, x_posts, P_posts,
                              P_priors, use_steady=use_steady)
        return X, P, J, lls
