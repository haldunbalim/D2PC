import argparse
import os
import numpy as np
from d2pc import *
from d2pc.experiment.figure import *
import pandas as pd

# args for true system, and estimation problem
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="Random Seed")
parser.add_argument("--wdir", type=str, default=None,
                    help="Folder name for the current experiment, if not provided retrieves last one in outputs folder")
parser.add_argument("--T", type=int, default=30, help="Prediction horizon")
parser.add_argument("--p-cc", type=float, default=.95,
                    help="Probability of chance constraint satisfaction")
parser.add_argument("--n-time",  type=int, default=10,
                    help="Number of samples for timing")
parser.add_argument("--recompute", action="store_true",
                    help="Recompute the results")
args = parser.parse_args()
set_seed(args.seed)

# load data
wdir, data = read_offline_data(args.wdir)
h = np.array([0, 0, 0, 0, 1])
true_sys = data["true_sys"]
orders = list(range(2, 11)) + [15]
logger = get_logger(os.path.join(wdir, "sddpc-compare"))
if not os.path.exists(os.path.join(wdir, "sddpc-compare.pkl")) or args.recompute:
    if not os.path.exists(os.path.join(wdir, "offline-arx.pkl")):
        raise Exception("Run the sprint_mass_offline_arx.py first!")
    with open(os.path.join(wdir, "offline-arx.pkl"), "rb") as f:
        arx_data = pickle.load(f)
        arx_sys, arx_ctrlr = arx_data["sys"], arx_data["ctrlr"]
        y_lim, u_lim = arx_data["y_lim"], arx_data["u_lim"]

    est_sys, ctrlr = data["est_sys"], data["ctrlr"]
    ys, us, ws = data["ys"], data["us"], data["ws"]
    P0 = true_sys.get_steady_post_covar()

    # recompute stoch tube for this experiment
    Hy = ctrlr.sys.C / y_lim
    Hu = np.eye(ctrlr.sys.nu) / u_lim
    H = block_diag(Hy, Hu)
    H = np.vstack([H, -H])
    ctrlr.H = H
    _, calXs = stoch_tube_design_time_varying(est_sys, ctrlr.dynof, 2,
                                              block_diag(P0, np.zeros_like(P0)), H,
                                              ctrlr.J, ctrlr.hess_th, args.p_cc)
    ctrlr.compute_constants()

    # compute tightening for D2PC
    tights_lmi, tights_soc, tights_true = sddpc_compare_d2pc(
        true_sys, est_sys, ctrlr, args.T, h, args.p_cc, P0)
    tights_lmi_arx, tights_soc_arx = sddpc_compare_d2pc_arx(
        arx_ctrlr, args.T, args.p_cc, h)
    # compute tightening for S-DDPC
    u_traj = np.ones((args.T, true_sys.nu))
    tight_smm = compute_tight_sddpc(true_sys, ys, us, u_traj, P0, h, args.T, args.p_cc,
                                    true_sys.Q, true_sys.R[0, 0], ws, orders)

    # timing
    P0y = true_sys.C @ P0 @ true_sys.C.T + true_sys.R
    y = np.random.multivariate_normal(np.zeros(ctrlr.sys.ny), P0y, max(orders))

    times = []
    for L0 in orders:
        prob, y0, u_cp = sddpc_prob(true_sys, ys, us, ws, P0, y_lim, u_lim, L0, args.T,
                                    args.p_cc, true_sys.Q, true_sys.R[0, 0], ctrlr.costR)
        y0.value = np.zeros(y0.shape)
        times.append(time_solve(prob, n_time=args.n_time, solver=cp.MOSEK))
    arx_ctrlr.T, ctrlr.T = args.T, args.T
    times.append(time_prob_d2pc(arx_ctrlr, x0=np.zeros(arx_ctrlr.sys.nx), approx=True, n_time=args.n_time))
    times.append(time_prob_d2pc(arx_ctrlr, x0=np.zeros(arx_ctrlr.sys.nx), approx=False, n_time=args.n_time))
    times.append(time_prob_d2pc(ctrlr, x0=np.zeros(ctrlr.sys.nx), approx=True, n_time=args.n_time))
    times.append(time_prob_d2pc(ctrlr, x0=np.zeros(ctrlr.sys.nx), approx=False, n_time=args.n_time))
    times = np.array(times)
    with open(os.path.join(wdir, "sddpc-compare.pkl"), "wb") as f:
        pickle.dump({"tight_smm": tight_smm, "orders": orders,
                        "tights_lmi": tights_lmi, "tights_soc": tights_soc,
                        "tights_lmi_arx": tights_lmi_arx, "tights_soc_arx": tights_soc_arx,
                        "tights_true": tights_true, "times": times}, f)
else:
    with open(os.path.join(wdir, "sddpc-compare.pkl"), "rb") as f:
        data = pickle.load(f)
        orders = data["orders"]
        tight_smm = data["tight_smm"]
        tights_lmi = data["tights_lmi"]
        tights_lmi_arx = data["tights_lmi_arx"]
        tights_soc_arx = data["tights_soc_arx"]
        tights_soc = data["tights_soc"]
        tights_true = data["tights_true"]
        times = data["times"]

tight_d2pc = np.array([tights_soc_arx, tights_lmi_arx,
                      tights_soc, tights_lmi])[..., -1]
tight_smm = tight_smm[:, -1]
set_fig_params()
plt.rcParams['xtick.labelsize'] = 18
fig = plt.figure(figsize=(9, 6), constrained_layout=True)
ax = fig.add_subplot(111)
std_mul = 3
ax.errorbar(range(tight_smm.shape[0]), np.mean(tight_smm, axis=-1) / tights_true[-1],
            yerr=np.std(tight_smm, axis=-1) / tights_true[-1] * std_mul, c="tab:blue", fmt="o", capsize=3)
ax.scatter(range(tight_smm.shape[0], tight_smm.shape[0]+2),  tight_d2pc[:2] / tights_true[-1], c="tab:cyan")
ax.scatter(range(tight_smm.shape[0]+2, tight_smm.shape[0]+4), tight_d2pc[2:] / tights_true[-1], c="tab:orange")
ax.scatter(tight_smm.shape[0]+4, 1, c="tab:green")

ax.set_ylabel("Tightening")
names = ["Direct order=" + str(o) for o in orders]
names += ["ARX-D2PC-SOC", "ARX-D2PC-LMI", "D2PC-SOC", "D2PC-LMI", "True"]
ax.set_xticks(range(len(names)), names, rotation=90)
ax.grid()
ax.set_yscale("log")
fig.savefig(os.path.join(wdir, "sddpc-compare.png"))

times_st = np.array([f"{np.mean(t):.3f} +- {np.std(t):.3f}" if np.all(t != -1) else "infeasible" for t in times])
df = pd.DataFrame({"Method": names[:-1], "Time (s)": times_st})
logger.info(str(df))