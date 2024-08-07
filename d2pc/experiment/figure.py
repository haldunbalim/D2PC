import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.stats import chi2, norm
from d2pc.utils import *
import os
import pickle
import glob
from typing import Any, Union, Optional
import logging
from copy import deepcopy

# ------ NOTE ------
# This script is primarily for generating data for the figures in the paper
# This file can be treated independently of the rest of the codebase for other purposes


def set_fig_params():
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 20


def read_offline_data(wdir: str) -> Union[str, dict[str, Any]]:
    """
        Read the offline data from the given folder name
            - wdir: the folder name
    """
    if wdir is None:
        assert os.path.exists("outputs"), "No output folder path provided and no outputs folder found."
        # load the latest data file if not provided
        wdirs = list(sorted(glob.glob("outputs/*")))
        assert len(wdirs) > 0, "No data folder found in outputs folder."
        wdir = wdirs[-1]
        assert os.path.exists(os.path.join(wdir, "offline.pkl")
                              ), f"No offline data found in the output folder: {wdir}"
    with open(os.path.join(wdir, "offline.pkl"), "rb") as f:
        data = pickle.load(f)
    return wdir, data


def shaded_plot(signal_mean: np.ndarray, signal_std: np.ndarray, 
                std_mul: Optional[int] = 3, color: Optional["str"] = None, ax: Optional[plt.Figure] = None,
                shade_alpha: Optional[float] = 0.5, plot_upside: Optional[bool] = True, plot_downside: Optional[bool] = True):
    """
        Plot the signal with shaded standard deviation
            - signal_mean: the mean of the signal 
            - signal_std: the standard deviation of the signal
            - std_mul: the multiplier for the standard deviation
            - color: the color of the plot
            - ax: the axis to plot on
            - shade_alpha: the alpha for the shaded region
            - plot_upside: whether to plot the upper side of the shaded region
            - plot_downside: whether to plot the lower side of the shaded region 
    """
    ax = ax or plt.gca()
    p = ax.plot(signal_mean, color=color)
    u = signal_mean + std_mul*signal_std if plot_upside else signal_mean
    d = signal_mean - std_mul*signal_std if plot_downside else signal_mean
    ax.fill_between(range(len(signal_mean)), u, d,
                    alpha=shade_alpha, color=p[0].get_color())
    ax.grid()

def simulate_ctrlr_multi(Nsim, true_sys, dynof, nus: np.ndarray,
                   init_mean: np.ndarray, init_covar: np.ndarray,
                   costQ: np.ndarray, costR: np.ndarray):
    """
        Simulate the true system with the controller multiple times
            - Nsim: int, the number of simulations
            - true_sys: LTISystem the true system
            - dynof: DynOFController, the dynamic output-feedback controller
            - nus: the controller input sequence
            - init_mean: the initial mean
            - init_covar: the initial covariance
            - costQ: the state cost
            - costR: the control cost
    """
    init_state = np.random.multivariate_normal(init_mean, init_covar, Nsim)
    xs = [init_state]
    ys = []
    us = []
    xcs = np.zeros((Nsim, true_sys.nx))
    for _nu in nus:
        # measure
        v = np.random.multivariate_normal(np.zeros(true_sys.ny), true_sys.R, size=Nsim)
        ys.append(xs[-1] @ true_sys.C.T + v)

        # control
        u_dynof = xcs @ dynof.K.T
        us.append(u_dynof + _nu)

        # dynamics
        w = np.random.multivariate_normal(np.zeros(true_sys.nw), true_sys.Q, size=Nsim)
        xs.append(xs[-1] @ true_sys.A.T + us[-1] @
                  true_sys.B.T + w @ true_sys.E.T)
        xcs = xcs @ dynof.Ac.T + ys[-1] @ dynof.L.T
    xs, ys, us = np.array(xs), np.array(ys), np.array(us)
    costs = np.sum(np.squeeze(xs[1:, :, None, :] @
                 costQ @ xs[1:, ..., None]), axis=0)
    costs += np.sum(np.squeeze(us[:, :, None, :] @
                  costR @ us[:, ..., None]), axis=0)
    
    xs = np.swapaxes(xs, 0, 1)
    ys = np.swapaxes(ys, 0, 1)
    us = np.swapaxes(us, 0, 1)
    
    return xs, ys, us, costs / len(nus)

def process_ctrlr_data(xs, costs, v_lim):
    xs_vio_mean = np.mean(xs > v_lim, axis=0)
    xs_mean = np.mean(xs, axis=0)
    xs_std = np.std(xs, axis=0)

    constr_vio = np.mean(xs[:, :, 1::2] > v_lim, axis=0)
    max_vio = np.max(constr_vio, axis=0)
    max_vio_channel = np.argmax(max_vio)
    max_vio = max_vio[max_vio_channel]
    max_vio_time = np.argmax(constr_vio[:, max_vio_channel])
    avg_cost = np.mean(costs)

    return xs_mean, xs_std, xs_vio_mean, max_vio, max_vio_time, max_vio_channel, avg_cost

def sample_calAB(ctrlr, num_samples, samples0=None):
    d_sqrt = np.sqrt(chi2.ppf(ctrlr.delta, ctrlr.hess_th.shape[0]))
    sys = ctrlr.sys
    # sample from \Theta_\delta
    th_samples = np.random.rand(num_samples, ctrlr.J.shape[1]) * 2 - 1
    if samples0 is not None:
        th_samples = np.concatenate([th_samples, samples0], axis=0)
    th_samples /= np.linalg.norm(th_samples, axis=1)[:, None]
    th_samples = th_samples @ la.sqrtm(ctrlr.covar_th) * d_sqrt
    Th_samples = th_samples @ ctrlr.J.T
    E_pinv_AB_samples = Th_samples.reshape(-1,
                                           sys.nw, sys.nx+sys.nu, order="F")
    AB_samples = sys.E @ E_pinv_AB_samples
    A_samples, B_samples = AB_samples[:, :, :sys.nx], AB_samples[:, :, sys.nx:]
    ABK_samples = np.concatenate(
        [A_samples, B_samples @ ctrlr.dynof.K], axis=-1)
    calA_samples = np.concatenate(
        [ABK_samples, np.zeros_like(ABK_samples)], axis=1)
    calA_samples += ctrlr.dynof.calA
    calBv_samples = np.concatenate([B_samples + sys.B, np.zeros_like(B_samples)], axis=1)
    return calA_samples, calBv_samples

def fwd_reach_nom_tube(ctrlr, nus, x0, init_tube_size=0):
    """ 
        Compute the forward reachable set predicted by SOC-based tube dynamics and LMI-based tube dynamics
    """
    # matrices
    sys = ctrlr.sys
    d_sqrt = np.sqrt(chi2.ppf(ctrlr.delta, ctrlr.J.shape[1]))
    calA = ctrlr.dynof.calA
    calBv = ctrlr.dynof.calBv
    nx, nw = sys.nx, sys.nw

    # required matrices for LMI
    covar_th_sqrt = la.sqrtm(ctrlr.covar_th)
    Bp = np.vstack([sys.E, np.zeros((nx, nw))])
    PBp = la.sqrtm(ctrlr.calP) @ Bp
    Sigma_J_sqrt = np.kron(np.eye(sys.nx+sys.nu), PBp) @ ctrlr.J @ covar_th_sqrt

    # required matrices for SOC
    Sigma_J = Sigma_J_sqrt @ Sigma_J_sqrt.T
    I = np.eye(2*nx)
    Ikes = [np.kron(np.eye(nx+sys.nu), I[i][:, None]) for i in range(2*nx)]
    Sigma_bar = np.sum([Ike.T @ Sigma_J @ Ike for Ike in Ikes], axis=0)
    Sigma_bar_sqrt = la.sqrtm(Sigma_bar)

    # init
    xi = np.concatenate([x0, np.zeros_like(x0)])
    tubes_lmi = [init_tube_size]
    tubes_soc = [init_tube_size]
    us = []
    # compute the tubes
    for _nu in nus:
        us.append(ctrlr.dynof.K @ xi[nx:] + _nu)
        xu = np.concatenate([xi[:nx], us[-1]])
        lmi_mat = np.kron(xu[None, :], np.eye(2*nx)) @ Sigma_J_sqrt
        offset_lmi = sigma_max(lmi_mat)
        tubes_lmi.append(ctrlr.rho * tubes_lmi[-1] + d_sqrt * offset_lmi)
        offset_soc = np.linalg.norm(Sigma_bar_sqrt @ xu)
        tubes_soc.append(ctrlr.rho * tubes_soc[-1] + d_sqrt * offset_soc)
        xi = calA @ xi + calBv @ _nu
    return np.array(tubes_lmi), np.array(tubes_soc), np.array(us)

def fwd_reach_nom_tube_sampling(ctrlr, nus, x0, num_samples=10000):
    sys = ctrlr.sys

    # first we estimate maximizer for tube size \foreach t \in \I{0, T} by maximizing lower bound
    calA, calBv = ctrlr.dynof.calA, ctrlr.dynof.calBv
    P_sqrt = la.sqrtm(ctrlr.calP)
    Bp = np.vstack([sys.E, np.zeros((sys.nx, sys.nw))])
    PBp = P_sqrt @ Bp
    covar_th_sqrt = la.sqrtm(ctrlr.covar_th)
    Sigma_J_sqrt = np.kron(np.eye(sys.nx+sys.nu), PBp) @ ctrlr.J @ covar_th_sqrt
    lmi_mats = []
    xi_mean = [np.concatenate([x0, np.zeros_like(x0)])]
    # Propogate xi_mean, also compute lmi_mats that give the tube size
    for _nu in nus:
        xu = np.concatenate(
            [xi_mean[-1][:sys.nx], ctrlr.dynof.K @ xi_mean[-1][sys.nx:] + _nu])
        lmi_mat = np.kron(xu[None, :], np.eye(2*sys.nx)) @ Sigma_J_sqrt
        lmi_mats.append(lmi_mat)
        xi_mean.append(calA@xi_mean[-1] + calBv@_nu)
    xi_mean = np.array(xi_mean)
    
    ehs = []
    # maximize convex lower bound for \alpha_T^2
    for i in range(1, len(nus)+1):
        curr_lmi_mats = lmi_mats[:i]
        rho_seq = np.array([ctrlr.rho**i for i in reversed(range(i))])
        curr_lmi_mats = (rho_seq**2)[:, None, None] * curr_lmi_mats
        curr_lmi_mat = np.sum(curr_lmi_mats, axis=0)
        curr_lmi_mat = curr_lmi_mat.T @ curr_lmi_mat
        eh = np.squeeze(la.eigh(curr_lmi_mat, subset_by_index=[
                        curr_lmi_mat.shape[0]-1, curr_lmi_mat.shape[0]-1])[1])
        ehs.append(eh)
    ehs = np.array(ehs)
    # add random sampled thetas
    calA_samples, calBv_samples = sample_calAB(ctrlr, num_samples - len(ehs), ehs)

    # propagate xis for each sample
    xis = [np.repeat(np.concatenate([x0, np.zeros_like(x0)])[None], calA_samples.shape[0], axis=0)]
    nx = sys.nx
    for _nu in nus:
        xis.append(np.squeeze(calA_samples @ xis[-1][..., None]) + calBv_samples @ _nu)
    # compute tube size for each
    xis = np.array(xis)
    xi_mean = np.array(xi_mean)
    xis_diff = xis - xi_mean[:, None]
    xiP_norms = np.sqrt(np.squeeze(xis_diff[..., None, :] @ ctrlr.calP @ xis_diff[..., None]))
    # return maximal tube size for each time-step
    return np.max(xiP_norms, axis=1)

def compute_constr_tight_sampling(ctrlr, sys, P0, T_err_plot=20, num_samples=10000):
    """
        Compute the constraint tightening over time using sampling from \Theta_\delta
    """
    c = norm.ppf(ctrlr.p_cc)
    calA_samples, _ = sample_calAB(ctrlr, num_samples)

    # compute the error covariance bound for each sample
    xi_bounds = [np.repeat(block_diag(P0, np.zeros_like(P0))[
        None], num_samples, axis=0)]
    BdBdT = block_diag(sys.E@sys.Q@sys.E.T,
                       ctrlr.dynof.L @ sys.R @ ctrlr.dynof.L.T)
    IK = block_diag(np.eye(sys.nx), ctrlr.dynof.K)
    for i in range(T_err_plot):
        xi_bounds.append(
            calA_samples @ xi_bounds[-1] @ np.transpose(calA_samples, (0, 2, 1)) + BdBdT)
    xi_bounds = np.array(xi_bounds)
    # compute the tightening for each sample
    stoch_tights_sampling = np.sqrt(
        np.array([h @ xi_bounds @ h for h in (ctrlr.H@IK)]))
    # compute the max tightening
    stoch_tights_sampling = np.max(stoch_tights_sampling, axis=-1) * c
    stoch_tights_sampling = stoch_tights_sampling.T

    # compute the error covariance bound with mean parameters
    mean_xi_bound = [block_diag(P0, np.zeros_like(P0))]
    for i in range(T_err_plot):
        mean_xi_bound.append(
            ctrlr.dynof.calA @ mean_xi_bound[-1] @ ctrlr.dynof.calA.T + BdBdT)
    mean_xi_bound = np.array(mean_xi_bound)
    # compute the tightening for mean parameters
    stoch_tights_mean = np.sqrt(
        np.array([h @ mean_xi_bound @ h for h in ctrlr.H@IK])).T * c
    return stoch_tights_sampling, stoch_tights_mean


def nom_tube_plot(tubes_lmi, tubes_soc, tubes_sampling):
    """
        Plot the tube sizes over time
    """
    fig = plt.figure(figsize=(9, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.set_ylabel(r"Tube size $\alpha$")
    ax.set_xlabel("time-step")
    ax.grid()
    ax.plot(tubes_lmi)
    ax.plot(tubes_soc, linestyle="dashed")
    ax.plot(tubes_sampling, linestyle="dotted")
    ax.legend(["LMI", "SOC", "Sampling"])
    return fig


def plot_constraint_tightening(stoch_tights_ctrlr, stoch_tights_sampling, stoch_tights_mean, mass_nr, t_beg=5):
    """
        Plot the constraint tightening over time
    """
    tv = stoch_tights_ctrlr
    inf = np.repeat(tv[-1][None, :],
                    len(stoch_tights_sampling) - len(tv), axis=0)
    ct_st = np.concatenate([tv, inf], axis=0)

    fig = plt.figure(figsize=(9, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.set_ylabel(f"Stochastic tightening on $x_"+"{"+ str(mass_nr*2+2) + "}$")
    ax.set_xlabel("time-step")
    ax.tick_params(axis='y', which='major', labelsize=16)
    ax.tick_params(axis='y', which='minor', labelsize=16)
    ax.tick_params(axis='x', which='major', labelsize=16)
    ax.tick_params(axis='x', which='minor', labelsize=16)
    ax.plot(range(t_beg, ct_st.shape[0]),
            ct_st[t_beg:, mass_nr], linewidth=2, c="tab:blue")
    ax.plot(range(t_beg, stoch_tights_mean.shape[0]),
            stoch_tights_mean[t_beg:, mass_nr], linewidth=2, c="tab:orange", linestyle="dashed")
    ax.plot(range(t_beg, stoch_tights_sampling.shape[0]),
            stoch_tights_sampling[t_beg:, mass_nr], linewidth=2, c="tab:green", linestyle="dotted")
    ax.legend(["Proposed", "Certainty Equivalent", "Sampling"])
    ax.grid()
    return fig


def control_plot(xs_mean_lmi: np.ndarray, xs_std_lmi: np.ndarray, xs_vio_mean_lmi: np.ndarray,
                 xs_mean_soc: np.ndarray, xs_std_soc: np.ndarray, xs_vio_mean_soc: np.ndarray,
                 v_lim: int, p_cc: float, mass_nr: int = 0, std_mul: int = 3):
    """
        Control figure
            Plot the position and velocity for mass with given number over time
            Plot the probability of velocity > v_lim over time
    """
    fig, axs = plt.subplots(3, 2, figsize=(6, 9), constrained_layout=True)
    for i, (xs_mean, xs_std, xs_vio_mean) in enumerate([(xs_mean_lmi, xs_std_lmi, xs_vio_mean_lmi), 
                                                       (xs_mean_soc, xs_std_soc, xs_vio_mean_soc)]):

        shaded_plot(xs_mean[:, mass_nr*2], xs_std[:, mass_nr*2], std_mul=std_mul, color="tab:blue",
                    ax=axs[0, i], shade_alpha=0.33)
        p, v = str(mass_nr*2+1), str(mass_nr*2+2)
        axs[0, i].set_ylabel("$x_{"+p+"}$")
        shaded_plot(xs_mean[:, mass_nr*2+1], xs_std[:, mass_nr*2+1], std_mul=std_mul, color="tab:blue",
                    ax=axs[1, i], shade_alpha=0.33)
        axs[1, i].set_ylabel("$x_{"+v+"}$")

        axs[1, i].plot([0, xs_mean.shape[0]], [v_lim, v_lim],
                       linestyle="--", color="tab:red")
        axs[2, i].plot([0, xs_mean.shape[0]], [1-p_cc, 1-p_cc],
                       linestyle="--", color="tab:red")
        axs[2, i].plot(xs_vio_mean[:, mass_nr*2+1])
        axs[2, i].grid()
        axs[2, i].set_ylabel(
            r"$\mathrm{Pr}$" + "$[x_{"+v+"}" + f"\geq {v_lim}]$")
        for j in range(3):
            axs[j, i].set_xlabel("time-step", fontsize=15)
    axs[0, 0].set_title("LMI Tube Dyn.")
    axs[0, 1].set_title("SOC Tube Dyn.")

    axs[2, 0].set_yticks([0, 0.02, 0.04], ["0%", "2%", "4%"])
    axs[2, 1].set_yticks([0, 0.02, 0.04], ["0%", "2%", "4%"])
    return fig

def get_logger(path, ext="log"):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.addHandler(logging.FileHandler(path+"."+ext, 'w'))
    return logger


def sddpc_compare_d2pc(true_sys, est_sys, ctrlr, T, h, p, P0):
    """
        Setup the data for the S-DDPC comparison
        - true_sys: the true system
        - est_sys: the estimated system
        - ctrlr: the controller
        - T: the prediction horizon
        - h: the constraint vector
        - p: the probability of chance constraint satisfaction
        - P0: the initial covariance
    """
    x0 = np.zeros(true_sys.nx)
    P0 = block_diag(P0, np.zeros_like(P0))
    # compute covariance bounds
    sigmas_true = [P0]
    dynof = deepcopy(ctrlr.dynof)
    dynof.sys = true_sys
    calA = dynof.calA
    BdBdT = block_diag(true_sys.E @ true_sys.Q @ true_sys.E.T,
                       dynof.L @ true_sys.R @ dynof.L.T)
    for _ in range(T):
        sigmas_true.append(calA @ sigmas_true[-1] @ calA.T + BdBdT)

    # compute the stochastic tightening terms
    stoch_tights = np.array([norm.ppf(p) * np.sqrt(h.T @ (est_sys.C @ calX[:est_sys.nx,
                            :est_sys.nx] @ est_sys.C.T + est_sys.R) @ h) for calX in ctrlr.sigma_tvs])
    stoch_tights = np.concatenate(
        [stoch_tights, [stoch_tights[-1]] * (T-len(stoch_tights)+1)])
    tights_true = np.array([norm.ppf(p) * np.sqrt(h.T @ (est_sys.C @ calX[:est_sys.nx,
                            :est_sys.nx] @ est_sys.C.T + est_sys.R) @ h) for calX in sigmas_true])

    tubes_lmi, tubes_soc = [], []
    calA, calBv = ctrlr.dynof.calA, ctrlr.dynof.calBv
    nus = []
    xi = np.concatenate([x0, np.zeros_like(x0)])
    for _ in range(T):
        nus.append(np.ones(est_sys.nu) - ctrlr.dynof.K @ xi[est_sys.nx:])
        xi = calA @ xi + calBv @ nus[-1]
    nus = np.array(nus)
    # nom tights
    tubes_lmi, tubes_soc, _ = fwd_reach_nom_tube(ctrlr, nus, x0=x0)
    # tights
    Pinv = la.inv(ctrlr.calP)
    f = np.sqrt(h.T @ est_sys.C @ Pinv[:est_sys.nx, :est_sys.nx] @ est_sys.C.T @ h)
    tights_lmi = (f * tubes_lmi + stoch_tights)
    tights_soc = (f * tubes_soc + stoch_tights)
    return tights_lmi, tights_soc, tights_true


def sddpc_compare_d2pc_arx(ctrlr, T, p, h):
    """
        Setup the data for the S-DDPC comparison for ARX models
        - ctrlr: the controller
        - T: the prediction horizon
        - p: the probability of chance constraint satisfaction
        - h: constraint vector
    """
    est_sys = ctrlr.sys
    x0 = np.zeros(est_sys.nx)
    stoch_tights = np.array([norm.ppf(p) * np.sqrt(h.T @ (est_sys.C @ calX[:est_sys.nx,
                            :est_sys.nx] @ est_sys.C.T + est_sys.R) @ h) for calX in ctrlr.sigma_tvs])
    stoch_tights = np.concatenate(
        [stoch_tights, [stoch_tights[-1]] * (T-len(stoch_tights)+1)])

    calA, calBv = ctrlr.dynof.calA, ctrlr.dynof.calBv
    nus = []
    xi = np.concatenate([x0, np.zeros_like(x0)])
    for _ in range(T):
        nus.append(np.ones(est_sys.nu) - ctrlr.dynof.K @ xi[est_sys.nx:])
        xi = calA @ xi + calBv @ nus[-1]
    nus = np.array(nus)
    tubes_lmi, tubes_soc, _ = fwd_reach_nom_tube(ctrlr, nus, x0=x0)

    Pinv = la.inv(ctrlr.calP)
    f = np.sqrt(h.T @ est_sys.C @
                Pinv[:est_sys.nx, :est_sys.nx] @ est_sys.C.T @ h)
    return f * tubes_lmi + stoch_tights, f * tubes_soc + stoch_tights


def compute_tight_sddpc(true_sys, ys, us, u_traj, P0, h, T, p, Q, r, ws, orders, n_ysamples=10000):
    """
        Compute the tightening for S-DDPC
        - true_sys: the true system
        - ys: the output sequence
        - us: the input sequence
        - u_traj: input trajectories
        - P0: the initial covariance
        - h: constraint vector
        - T: the prediction horizon
        - p: the probability of chance constraint satisfaction
        - Q: disturbance cov
        - r: measurement noise cov
        - ws: the disturbance sequence
        - orders: the orders to consider
        - n_ysamples: the number of samples to average over
    """

    tight_smm = []
    for L0 in orders:
        # construct matrices
        L = L0 + T
        M = len(ys)-L
        U = np.array([la.hankel(_us[:L], _us[L-1:]) for _us in us[1:].T])
        U = np.transpose(U, (1, 0, 2)).reshape(-1, U.shape[-1])

        W = np.array([la.hankel(_ws[:L], _ws[L-1:]) for _ws in ws[1:].T])
        W = np.transpose(W, (1, 0, 2)).reshape(-1, W.shape[-1])
        Psi = np.vstack([U, W])

        Y = np.array([la.hankel(_ys[:L], _ys[L-1:]) for _ys in ys[:-1].T])
        Y = np.transpose(Y, (1, 0, 2)).reshape(-1, Y.shape[-1])
        Yp, Yf = Y[:L0*true_sys.ny], Y[L0*true_sys.ny:]

        # compute gamma estimate
        F = L * r * np.eye(M) + Yp.T @ Yp
        Finv = psd_inverse(F)
        smm_gamma = Yf @ (Finv - Finv @ Psi.T @ psd_inverse(Psi @ Finv @ Psi.T) @ Psi @ Finv) @ Yp.T

        QQ = smm_gamma.T @ smm_gamma
        _lambda = r * (np.trace(QQ)+T)
        F = _lambda * np.eye(M) + Yp.T @ QQ @ Yp

        # compute tightening terms
        FF2 = Finv @ Psi.T @ psd_inverse(Psi @ Finv @ Psi.T)
        FF1 = (Finv - FF2 @ Psi @ Finv) @ Yp.T @ QQ
        Gamw = (Yf - smm_gamma @ Yp) @ FF2[:, L*true_sys.nu:]
        Sigw = block_diag(Q, L)

        P0_smm = true_sys.C @ P0 @ true_sys.C.T + np.eye(true_sys.ny) * r
        barH = block_diag(h, T)
        cov = smm_gamma @ block_diag(P0_smm,
                                     L0) @ smm_gamma.T + Gamw @ Sigw @ Gamw.T
        FF3 = np.sqrt(np.diag(barH @ cov @ barH.T))
        FF4 = np.sqrt(np.diag(
            barH @ (r * (smm_gamma @ smm_gamma.T + np.eye(T * true_sys.ny))) @ barH.T))
        u_term = u_traj.flatten() @ FF2[:, L0*true_sys.nu: L*true_sys.nu].T
        
        # generate y-sequence
        _x0 = [np.random.multivariate_normal(np.zeros(true_sys.nx), P0, n_ysamples)]
        for _ in range(L0-1):
            w = np.random.multivariate_normal(np.zeros(true_sys.nw), true_sys.Q, n_ysamples) 
            _x0.append(_x0[-1] @ true_sys.A.T + w @ true_sys.E.T)
        _x0 = np.array(_x0)
        _y0 = (true_sys.C @ _x0[..., None])[..., 0]
        _y0 += np.random.multivariate_normal(np.zeros(true_sys.ny), true_sys.R, n_ysamples)
        y0 = np.transpose(_y0, (1, 0, 2))

        #y0 = np.random.multivariate_normal(np.zeros(true_sys.ny), P0_smm, (n_ysamples, L0))
        gs = (y0.reshape(n_ysamples, -1) @ FF1.T)[:, None] + u_term[None]

        tights = FF4[..., None, None] * \
            np.linalg.norm(gs, axis=-1) + FF3[..., None, None]
        tights = np.concatenate(
            [np.zeros((1, *tights.shape[1:])), tights], axis=0)
        tights[0, ...] = np.sqrt(h.T @ P0_smm @ h)
        tight_smm.append(norm.ppf(p) * tights)
    tight_smm = np.array(tight_smm)[..., -1]
    return tight_smm


def sddpc_prob(true_sys, ys, us, ws, P0, y_lim, u_lim, L0, T, p, Q, r, costR):
    L = L0 + T
    M = len(ys)-L
    U = np.array([la.hankel(_us[:L], _us[L-1:]) for _us in us[1:].T])
    U = np.transpose(U, (1, 0, 2)).reshape(-1, U.shape[-1])

    W = np.array([la.hankel(_ws[:L], _ws[L-1:]) for _ws in ws[1:].T])
    W = np.transpose(W, (1, 0, 2)).reshape(-1, W.shape[-1])
    Psi = np.vstack([U, W])

    Y = np.array([la.hankel(_ys[:L], _ys[L-1:]) for _ys in ys[:-1].T])
    Y = np.transpose(Y, (1, 0, 2)).reshape(-1, Y.shape[-1])
    Yp, Yf = Y[:L0 * true_sys.ny], Y[L0 * true_sys.ny:]

    # compute gamma estimate
    F = L * r * np.eye(M) + Yp.T @ Yp
    Finv = psd_inverse(F)
    smm_gamma = Yf @ (Finv - Finv @ Psi.T @ psd_inverse(Psi @ Finv @ Psi.T) @ Psi @ Finv) @ Yp.T

    QQ = smm_gamma.T @ smm_gamma
    _lambda = r * (np.trace(QQ)+T)
    F = _lambda * np.eye(M) + Yp.T @ QQ @ Yp

    # compute tightening terms
    FF2 = Finv @ Psi.T @ psd_inverse(Psi @ Finv @ Psi.T)
    FF1 = (Finv - FF2 @ Psi @ Finv) @ Yp.T @ QQ
    Gamw = (Yf - smm_gamma @ Yp) @ FF2[:, L * true_sys.nu:]
    Sigw = block_diag(Q, L)

    P0_smm = true_sys.C @ P0 @ true_sys.C.T + np.eye(true_sys.ny) * r
    cov = smm_gamma @ block_diag(P0_smm,
                                 L0) @ smm_gamma.T + Gamw @ Sigw @ Gamw.T
    u_cp = cp.Variable((T * true_sys.nu))
    u0 = np.zeros((L0 * true_sys.nu))
    u_term = FF2[:, :-L * true_sys.nw] @ cp.hstack([u0, u_cp])

    y0 = cp.Parameter(L0 * true_sys.ny)
    g = FF1 @ y0 + u_term
    ydb = Yf @ g - smm_gamma @ (Yp @ g - y0)
    cost = cp.sum_squares(ys) + cp.sum_squares(block_diag(costR, T) @ u_cp)
    constraints = [u_term <= u_lim, u_term >= -u_lim]
    H = np.eye(ys.shape[1])
    for h in H:
        barH = block_diag(h / y_lim, T)
        FF3 = norm.ppf(p) * np.sqrt(np.diag(barH @ cov @ barH.T))
        FF4 = norm.ppf(p) * np.sqrt(np.diag(barH @ (r * (smm_gamma @
                                                         smm_gamma.T + np.eye(T * true_sys.nu))) @ barH.T))
        constraints.append(barH @ ydb + FF4 * cp.norm(g) + FF3 <= 1)
        constraints.append(-barH @ ydb + FF4 * cp.norm(g) + FF3 <= 1)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    return prob, y0, u_cp

def time_solve(prob, n_time=10, solver=cp.ECOS, ignore_first=True):
    if ignore_first:
        prob.solve(solver=solver)
    ls = []
    for _ in range(n_time):
        with timer() as t:
            prob.solve(solver=solver)
        ls.append(t() if prob.status == "optimal" else -1)
    return ls

def time_prob_d2pc(ctrlr, x0, approx, n_time, ignore_first=True):
    prob, (init_xi, init_alpha, stoch_tight), _ = ctrlr._construct_control_prob(approx)
    init_xi.value = np.hstack([x0, np.zeros_like(x0)])
    init_alpha.value = 0
    stoch_tight.value = np.array([ctrlr.stoch_tight_tv[0]] * ctrlr.T)
    solver = cp.ECOS if approx else cp.MOSEK
    return time_solve(prob, n_time=n_time, solver=solver, ignore_first=ignore_first)
