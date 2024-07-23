import argparse
import numpy as np
from d2pc import *
from d2pc.experiment.figure import *
import pandas as pd


# args for true system, and estimation problem
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=str, default=0, help="Random Seed")
parser.add_argument("--wdir", type=str, default=None,
                    help="Folder name for the current experiment, if not provided retrieves last one in outputs folder")
parser.add_argument("--num-val", type=int, default=100,
                    help="Number of validation sequnces")
parser.add_argument("--recompute", action="store_true",
                    help="Recompute the results")
args = parser.parse_args()
set_seed(args.seed)
# load data
orders = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
N_ls = len(orders)
wdir, data = read_offline_data(args.wdir)
logger = get_logger(os.path.join(wdir, "pred-err"))
true_sys = data["true_sys"]
if not os.path.exists(os.path.join(wdir, "pred-err.pkl")) or args.recompute:
    args_offline, est_sys, ctrlr = data["args"], data["est_sys"], data["ctrlr"]
    ys, us = data["ys"], data["us"]

    ths = []
    logger.info("Computing LS estimates for different orders")
    for order in orders:
        reg, label = setup_ls_arx(order, ys, us)
        th = la.lstsq(reg, label, cond=-1)[0].T
        ths.append(th)

    if not os.path.exists(os.path.join(wdir, "offline-lti.pkl")):
        raise Exception("Run spring_mass_offline_lti.py first")
    with open(os.path.join(wdir, "offline-lti.pkl"), "rb") as f:
        data = pickle.load(f)
        est_sys_lti = data["sys"]

    if not os.path.exists(os.path.join(wdir, "offline-arx.pkl")):
        raise Exception("Run spring_mass_offline_arx.py first")
    with open(os.path.join(wdir, "offline-arx.pkl"), "rb") as f:
        data = pickle.load(f)
        est_sys_unstructured = data["sys"]

    logger.info("Computing prediction errors for different models")
    errs = []
    for i in range(args.num_val):
        _errs = []
        _, val_ys, val_us = true_sys.simulate(
            args_offline.T, RandomNormalController(est_sys.nu, sigma_u=args_offline.sigma_u))
        for order, th in zip(orders, ths):
            _val_ys = np.vstack([np.zeros((order, val_ys.shape[1])), val_ys])
            _val_us = np.vstack([np.zeros((order, val_us.shape[1])), val_us])
            val_reg, val_label = setup_ls_arx(order, _val_ys, _val_us)
            es = val_label - val_reg @ th.T
            _errs.append(np.mean(np.sum(es**2, axis=1)))

        es = est_sys_lti.kf_fwd(val_ys, val_us, return_aux=True)[5]
        _errs.append(np.mean(np.sum(es**2, axis=1)))

        es = est_sys_unstructured.kf_fwd(val_ys, val_us, return_aux=True)[5]
        _errs.append(np.mean(np.sum(es**2, axis=1)))

        es = est_sys.kf_fwd(val_ys, val_us, return_aux=True)[5]
        _errs.append(np.mean(np.sum(es**2, axis=1)))

        es = true_sys.kf_fwd(val_ys, val_us, return_aux=True)[5]
        _errs.append(np.mean(np.sum(es**2, axis=1)))

        errs.append(_errs)

    errs_mean = np.mean(errs, axis=0)
    errs_std = np.std(errs, axis=0)
    with open(os.path.join(wdir, "pred-err.pkl"), "wb") as f:
        pickle.dump({"orders": orders, "errs_mean": errs_mean, "errs_std": errs_std, "ths": ths}, f)
else:
    with open(os.path.join(wdir, "pred-err.pkl"), "rb") as f:
        data = pickle.load(f)
        orders, errs_mean, errs_std = data["orders"], data["errs_mean"], data["errs_std"]

errs_std /= errs_mean[-1]
errs_mean /= errs_mean[-1]
names = ["LS order=" +str(o) for o in orders]
names += ["Full Param.", "ARX-GEM", "Str.-GEM", "True system"]
names = np.array(names)
df = pd.DataFrame(data=errs_mean, index=names, columns=["MSE"])
pd.set_option('display.float_format', lambda st: f"{st:.7f}")
logger.info(df)

set_fig_params()
plt.rcParams['xtick.labelsize'] = 18
fig = plt.figure(figsize=(9, 6), constrained_layout=True)
ax = fig.add_subplot(111)

std_mul = 3
cs = ["tab:blue"] * N_ls + ["tab:orange", "tab:red", "tab:cyan", "tab:green"]
for i, (err_mean, err_std, c) in enumerate(zip(errs_mean, errs_std, cs)):
    ax.errorbar(i, err_mean, yerr=err_std * std_mul, fmt="o", capsize=3, c=c)
ax.grid()
ax.set_ylabel("Prediction error")
ax.set_xticks(range(len(names)), names, rotation=90)
fig.savefig(os.path.join(wdir, "pred-err.png"), bbox_inches="tight")
