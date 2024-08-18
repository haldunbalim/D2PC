import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from d2pc import *
import pickle
from d2pc.experiment.figure import *
from tqdm import tqdm

# args for true system, and estimation problem
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="Random Seed")
parser.add_argument("--wdir", type=str, default=None,
                    help="Folder name for the current experiment, if not provided retrieves last one in outputs folder")
parser.add_argument("--num-trials", type=int, default=1000,
                    help="Number of trials to run")
parser.add_argument("--recompute", action="store_true",
                    help="Recompute the results")
args = parser.parse_args()
set_seed(args.seed)
# load data
wdir, data = read_offline_data(args.wdir)
logger = get_logger(os.path.join(wdir, "unc-quant"))

if not os.path.exists(os.path.join(wdir, "unc-quant.pkl")) or args.recompute:
    args_offline, true_sys = data["args"], data["true_sys"]
    num_masses, num_actuated = args_offline.num_masses, args_offline.num_actuated
    J, th0, true_vec = data["J"], data["th0"], data["true_vec"]

    syss, hess_ths, delta_mins = [], [], []
    AB_known = true_sys.E_ann.T @ np.hstack([true_sys.A, true_sys.B])
    for i in tqdm(range(args.num_trials)):
        xs, ys, us = true_sys.simulate(args_offline.T, RandomNormalController(num_actuated, sigma_u=args_offline.sigma_u))
        # create a random system as initial guess
        nx = num_masses * 2
        rand_vec = np.random.rand(J.shape[1]) * 2 + .5
        E_pinv_AB = (J @ rand_vec + th0).reshape(num_masses, -1, order="F")
        AB = true_sys.E @ E_pinv_AB + true_sys.E_ann_pinv.T @ AB_known
        rand_sys = LTISystem(A=AB[:, :nx], B=AB[:, nx:], C=true_sys.C, E=true_sys.E)
        rand_sys.P0 = np.eye(nx) * np.random.rand() * 1e-3
        Qprojectors, Qtypes = [np.eye(num_masses)], ["scaled"]
        rand_sys.Q = np.eye(num_masses) * np.random.rand() * 1e-3
        Rprojectors, Rtypes = [np.eye(num_masses)], ["scaled"]
        rand_sys.R = np.eye(num_masses) * np.random.rand() * 1e-3

        # setup the estimation problem
        setup = EstimationSetup(rand_sys, Qprojectors, Qtypes,
                                Rprojectors, Rtypes, J, th0)
        opt_problem = StructuredEMRunner(setup)

        # estimate the system
        est_sys, _ = opt_problem.em(ys, us, max_iter=2000, rtol=1e-7, verbose=0)

        # quantify the parameteric uncertainty
        pvec = (true_sys.E_pinv @ np.hstack([est_sys.A, est_sys.B])).ravel("F") - th0
        est_vec = la.pinv(J) @ pvec
        hess_th = get_nll_hess(est_sys, J, th0, ys, us)

        err_vec = est_vec - true_vec
        delta_min = chi2.cdf(err_vec.T @ hess_th @ err_vec, hess_th.shape[0])

        syss.append(est_sys)
        hess_ths.append(hess_th)
        delta_mins.append(delta_min)
    delta_mins = np.array(delta_mins)
    with open(os.path.join(wdir, "unc-quant.pkl"), "wb") as f:
        pickle.dump({"syss": syss, "hess_ths": hess_ths, "delta_mins": delta_mins}, f)
else:
    with open(os.path.join(wdir, "unc-quant.pkl"), "rb") as f:
        data = pickle.load(f)
        delta_mins = data["delta_mins"]

# plot the results
fig = plt.figure(figsize=(6, 6), constrained_layout=True)
ax = fig.add_subplot(111)
x = np.linspace(0, 1, 100)
y = []
for _x in x:
    y.append(np.mean(delta_mins < _x))
ax.plot(x, y)
ax.grid()
ax.plot([0, 1], [0, 1], c="black", linestyle="--")
fig.savefig(os.path.join(wdir, "unc-quant.png"))
for e in list(np.arange(.5, 1, .05)) + [.99, .999]:
    logger.info(f"Probability of true system being inside Theta_{e:.3f} is {np.mean(delta_mins <= e):.3f}")
