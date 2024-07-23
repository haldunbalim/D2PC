import argparse
from datetime import datetime
import os
import numpy as np
import scipy.linalg as la
from d2pc import *
from d2pc.experiment.figure import *
import pickle

# args for true system, and estimation problem
parser = argparse.ArgumentParser()
# system args
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--wdir", type=str, default=None,
                    help="Folder name for the current experiment, if not provided uses timestamp")
# args for the spring mass system
parser.add_argument("--num-masses", type=int, default=5, help="Number of masses in the system")
parser.add_argument("--num-actuated", type=int, default=5, help="Number of actuated masses")
parser.add_argument("-ts", type=float, default=.1, help="Sampling time")
# args for the estimation
parser.add_argument("-T", type=int, default=2000, help="Estimation sequence length")
parser.add_argument("--sigma-u", type=float, default=2, help="Excitation sequence standard deviation")
# args for the noise covariances
parser.add_argument("-q", type=float, default=3e-4, help="Process noise covariance")
parser.add_argument("-r", type=float, default=3e-4, help="Measurement noise covariance")
parser.add_argument("-p0", type=float, default=3e-4, help="Initial state covariance")
# args for the controller synthesis
parser.add_argument("--T-err", type=int, default=20, help="Horizon for error covariance bound")
parser.add_argument("--delta", type=float, default=.95, help="Probability level for considered parametric uncertainty")
parser.add_argument("--p-cc", type=float, default=.95, help="Probability of chance constraint satisfaction")
# args for the control problem
parser.add_argument("--v-lim", type=float, default=.3, help="Velocity limit")
parser.add_argument("--u-lim", type=float, default=3.5, help="Input limit")
parser.add_argument("--recompute", action="store_true", help="Recompute the results")

# retrieve args
args = parser.parse_args()
wdir = args.wdir 
if wdir is None:
    wdir = os.path.join("outputs", datetime.now().strftime('%Y%m%d%H%M%S'))
if os.path.exists(os.path.join(wdir, "offline.pkl")):
    if args.recompute:
        log_info("Results already computed, but re-computing...")
    else:
        log_info("Results already computed, exiting, give --recompute to re-run")
        exit(0)
set_seed(args.seed)
logger = get_logger(os.path.join(wdir, "offline"))
T = args.T
q, r, p0 = args.q, args.r, args.p0
num_masses = args.num_masses
num_actuated = args.num_actuated

# create the true system
ms = [np.random.uniform(.9, 1.1) for _ in range(num_masses)]
ks = [np.random.uniform(1.8, 2.2) for _ in range(num_masses)]
ds = [np.random.uniform(.9, 1.1) for _ in range(num_masses)]
A, B, C, E, J, th0 = create_spring_mass_sys(ms, ks=ks, ds=ds, num_actuated=num_actuated, ts=args.ts)
Q = np.eye(num_masses) * q
R = np.eye(num_masses) * r
P0 = np.eye(num_masses*2) * p0
true_sys = LTISystem(A=A, B=B, C=C, E=E, Q=Q, R=R, P0=P0)

# simulate the true system with random inputs
xs, ys, us, ws, vs = true_sys.simulate(T, RandomNormalController(num_actuated, sigma_u=args.sigma_u), return_noises=True)

# create a random system as initial guess
nx = num_masses * 2
rand_vec = np.random.rand(J.shape[1]) * 2 + .5
E_pinv_AB = (J @ rand_vec + th0).reshape(num_masses, -1, order="F")
AB_known = true_sys.E_ann.T @ np.hstack([true_sys.A, true_sys.B])
AB = E @ E_pinv_AB + true_sys.E_ann_pinv.T @ AB_known
rand_sys = LTISystem(A=AB[:, :nx], B=AB[:, nx:], C=C, E=E)
rand_sys.P0 = np.eye(nx) * np.random.rand() * 1e-3
Qprojectors, Qtypes = [np.eye(num_masses)], ["scaled"]
rand_sys.Q = np.eye(num_masses) * np.random.rand() * 1e-3
Rprojectors, Rtypes = [np.eye(num_masses)], ["scaled"]
rand_sys.R = np.eye(num_masses) * np.random.rand() * 1e-3

# cost matrices for the controller
costQ = np.diag([1, 0]*(true_sys.nx//2))
costR = np.eye(true_sys.nu) * 1e-4

# Form constraint matrix for input and state
u_lim, v_lim = args.u_lim, args.v_lim
Hx = np.eye(true_sys.nx)[1::2] / v_lim
Hx = np.vstack([Hx, -Hx])
Hu = np.eye(true_sys.nu) / u_lim
Hu = np.vstack([Hu, -Hu])
H = block_diag(Hx, Hu)

# init mean and covariance for the state for the control task
init_covar = np.eye(nx) * 1e-6

# setup the estimation problem
setup = EstimationSetup(rand_sys, Qprojectors, Qtypes,
                        Rprojectors, Rtypes, J, th0)
opt_problem = StructuredEMRunner(setup)

# estimate the system
with timer() as t:
    est_sys, _ = opt_problem.em(
        ys, us, max_iter=1000, rtol=1e-7, verbose=0)
    if logger is not None:
        logger.info("Estimation time: " + f"{t():.3f}"+"s")

# D2PC
ctrlr = d2pc_pipeline(est_sys, ys, us, J, th0, args.p_cc, args.delta, 
                      costQ, costR, H, init_covar, args.T_err, logger)

# log out if required delta min so that true system is inside \Theta_\delta, note in practice we dont know this value!
pvec = (est_sys.E_pinv @ np.hstack([est_sys.A, est_sys.B])).ravel("F") - th0
est_vec = la.pinv(J) @ pvec
true_vec = spring_mass_vec_fn(ms, ks=ks, ds=ds, num_actuated=num_actuated)
err_vec = est_vec - true_vec
delta_min = chi2.cdf(err_vec.T @ ctrlr.hess_th @ err_vec, ctrlr.hess_th.shape[0])
logger.info(f"Delta min: {delta_min:.3f}")
if delta_min > args.delta:
    logger.warning(
        "True system is not inside the uncertainty set, controller may not be robust")

# save the results
os.makedirs(wdir, exist_ok=True)
with open(os.path.join(wdir, "offline.pkl"), "wb") as f:
    pickle.dump({"args":args, "true_sys": true_sys, "est_sys": est_sys, "ctrlr": ctrlr,
                  "xs":xs, "ys":ys, "us": us, "ws":ws, "vs":vs, "init_covar": init_covar,
                  "ms":ms, "ks":ks, "ds":ds, "true_vec": true_vec, "J":J, "th0":th0}, f)
logger.info("Saved the results to: "+ wdir)
