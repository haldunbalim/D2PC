import argparse
import os
import numpy as np
from d2pc import *
from d2pc.experiment.figure import *
import pickle

# args for true system, and estimation problem
parser = argparse.ArgumentParser()
parser.add_argument("--wdir", type=str, default=None,
                    help="Folder name for the current experiment, if not provided retrieves last one in outputs folder")
parser.add_argument("--recompute", action="store_true",
                    help="Recompute the results")
args = parser.parse_args()
# load data
wdir, data = read_offline_data(args.wdir)
deltas = np.concatenate([np.linspace(.8, .98, 75), np.linspace(.98, 1-1e-16, 25)])
logger = get_logger(os.path.join(wdir, "dynof"))

if not os.path.exists(os.path.join(wdir, "dynof.pkl")) or args.recompute:
    args_offline, true_sys, est_sys, ctrlr = data["args"], data["true_sys"], data["est_sys"], data["ctrlr"]
    objs, oa_objs = [], []
    with timer() as t:
        dynof = synth_dynof_ctrl(est_sys, ctrlr.costQ,  ctrlr.costR)
        obj_nom, _ = an_dynof_ctrl(est_sys, dynof, ctrlr.costQ,  ctrlr.costR)
        time_nom = t()

    with timer() as t:
        for _delta in deltas:
            objs.append(synth_robust_dynof_ctrl(est_sys, ctrlr.J,
                        ctrlr.hess_th, _delta, ctrlr.costQ, ctrlr.costR, Lambda0_or_Ctrl0=dynof)[0])
        avg_time_dynof = t()/len(deltas)

    with timer() as t:
        D = optimize_D(ctrlr.J, ctrlr.covar_th, est_sys.nw)
        time_D = t()
    with timer() as t:
        for _delta in deltas:
            oa_objs.append(synth_robust_dynof_ctrl(
                est_sys, ctrlr.J, ctrlr.hess_th, _delta, ctrlr.costQ, ctrlr.costR, D=D, Lambda0_or_Ctrl0=dynof)[0])
        avg_time_dynof_approx = t()/len(deltas)

    with open(os.path.join(wdir, "dynof.pkl"), "wb") as f:
        pickle.dump({"deltas": deltas, "objs": objs, "oa_objs": oa_objs, "obj_nom": obj_nom,
                     "avg_time_dynof": avg_time_dynof, "time_D": time_D,
                     "avg_time_dynof_approx": avg_time_dynof_approx, "time_nom":time_nom}, f)
else:
    with open(os.path.join(wdir, "dynof.pkl"), "rb") as f:
        data = pickle.load(f)
        deltas, objs, oa_objs = data["deltas"], data["objs"], data["oa_objs"]
        obj_nom = data["obj_nom"]
        avg_time_dynof = data["avg_time_dynof"]
        time_D = data["time_D"]
        avg_time_dynof_approx = data["avg_time_dynof_approx"]
        time_nom = data["time_nom"]


logger.info("Time required for computing nominal LQG solution: " +
            f"{time_nom:.3f}"+" s")
logger.info("Average time required for synthesizing output-feedback controller: " +
            f"{avg_time_dynof:.3f}"+" s")
logger.info("Time required finding the shape matrix D for approximate set: " +
            f"{time_D:.3f}"+" s")
logger.info("Average time required for synthesizing output-feedback controller with approximate set: " +
            f"{avg_time_dynof_approx:.3f}"+" s")

set_fig_params()
fig = plt.figure(figsize=(9, 6), constrained_layout=True)
ax = fig.add_subplot(111)
ax.plot(deltas*100, objs / obj_nom, label="Full-block S-procedure", alpha=0.8, linewidth=2)
ax.plot(deltas*100, oa_objs / obj_nom, label="Over-approximate Set",
        alpha=0.8, linewidth=2, linestyle="dashed")
ax.plot([min(deltas)*100, 100], [1, 1], label="Nominal", c="black", linestyle="dotted")
ax.set_ylabel(r"$\mathcal{H}_2$-norm")
ax.set_xlabel(r"Probability Level $\delta$ (%)")
ax.legend()
ax.grid()
fig.savefig(os.path.join(wdir, "dynof-h2.png"))
