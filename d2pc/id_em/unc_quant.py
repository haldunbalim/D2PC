import numpy as np
import casadi as ca
import scipy.linalg as la

def kf_update_csadi(C, R, y, x_prior, P_t_tm1):
    ny, nx = C.shape
    S_t = C @ P_t_tm1 @ C.T + R
    S_t_chol = ca.chol(S_t)
    S_t_chol_inv = ca.solve(S_t_chol, ca.SX.eye(S_t_chol.shape[0]))
    S_t_inv = S_t_chol_inv@S_t_chol_inv.T
    K_t = P_t_tm1 @ C.T @ S_t_inv
    e = (y - C @ x_prior)
    x_t_t = x_prior + K_t @ e
    P_t_t = (np.eye(nx) - K_t @ C) @ P_t_tm1
    ll = 2 * ca.sum1(ca.log(ca.diag(S_t_chol)))
    ll += e.T @ S_t_inv @ e
    ll += ny * np.log(2*np.pi)
    return x_t_t, P_t_t, -0.5 * ll


def kf_predict_csadi(A, B, Q, x_t_t, P_t_t, u):
    P_t_tm1 = A @ P_t_t @ A.T + Q
    x_t_tm1 = A @ x_t_t + B @ u
    return x_t_tm1, P_t_tm1


def steady_filter_csadi(ys, us, x_post, A, B, C, R, P_prior_steady):
    nx, ny = x_post.shape[1], ys.shape[1]
    S_steady = C @ P_prior_steady @ C.T + R
    S_steady_chol = ca.chol(S_steady)
    S_steady_chol_inv = ca.solve(S_steady_chol, ca.SX.eye(S_steady_chol.shape[0]))
    S_inv_steady = S_steady_chol_inv @ S_steady_chol_inv.T
    logdetSt = 2 * ca.sum1(ca.log(ca.diag(S_steady_chol)))
    K_steady = (S_inv_steady @ C @ P_prior_steady).T

    x_priors = []
    x_posts = [x_post]
    x_posts[0] = x_post
    ll = 0
    for t, (y, u) in enumerate(zip(ys, us)):
        # predict
        x_priors.append(A @ x_posts[t] + B @ u)

        # update
        e = y - C @ x_priors[t]
        x_posts.append(x_priors[t] + K_steady @ e)
        ll += -.5 * (e.T @ S_inv_steady @ e + logdetSt + ny*np.log(2*np.pi))

    return ll


def apply_kf_csadi(A, B, C, E, Q, R, mu0, P0, ys, us, steady_idx=-1):
    x_priors = []
    P_priors = []
    x_posts = [mu0]
    P_posts = [P0]
    ll_tot = 0
    Q = E @ Q @ E.T
    for t, (y, u) in enumerate(zip(ys, us)):

        # predict
        x_t_tm1, P_prior = kf_predict_csadi(A, B, Q, x_posts[-1], P_posts[-1], u)

        # update
        x_t_t, P_post, ll = kf_update_csadi(C, R, y, x_t_tm1,  P_prior)

        # save
        P_posts.append(P_post)
        P_priors.append(P_prior)
        x_priors.append(x_t_tm1)
        x_posts.append(x_t_t)
        ll_tot += ll

        if steady_idx == t:
            ll_st = steady_filter_csadi(ys[t+1:], us[t+1:], x_posts[-1], A, B, C, R, P_priors[-1])
            ll_tot += ll_st
            break

    return ll_tot


def get_nll_hess(lti_sys, J, th0, ys, us, use_steady=True):
    nx, nw = lti_sys.nx, lti_sys.nw
    pvec = (lti_sys.E_pinv @ np.hstack([lti_sys.A, lti_sys.B])).ravel("F") - th0
    est_vec = la.pinv(J) @ pvec
    est_vec_cs = ca.SX.sym("AB", len(est_vec))
    E_pinv_AB = ca.reshape(J @ est_vec_cs + th0, nw, -1)
    AB = np.hstack([lti_sys.A, lti_sys.B])
    AB = lti_sys.E @ E_pinv_AB + lti_sys.E_ann_pinv.T @ lti_sys.E_ann.T @ AB
    steady_idx = lti_sys.get_kf_fwd_steady_idx() if use_steady else -1
    ll_tot = apply_kf_csadi(AB[:, :nx], AB[:, nx:], lti_sys.C, lti_sys.E, lti_sys.Q,
                            lti_sys.R, lti_sys.mu0, lti_sys.P0, ys, us, steady_idx=steady_idx)
    H, g = ca.hessian(-ll_tot, est_vec_cs)
    hess_th = np.array(ca.Function("somefn", [est_vec_cs], [H])(est_vec))
    return hess_th
