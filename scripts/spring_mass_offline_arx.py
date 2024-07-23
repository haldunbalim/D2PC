import argparse
import os
import numpy as np
from d2pc import *
from d2pc.experiment.figure import *

# args for true system, and estimation problem
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=str, default=0, help="Random Seed")
parser.add_argument("--wdir", type=str, default=None,
                    help="Folder name for the current experiment, if not provided retrieves last one in outputs folder")
parser.add_argument("--order", type=int, default=2, help="Control horizon")
parser.add_argument("--T-err", type=int, default=2,
                    help="Horizon for error covariance bound")
parser.add_argument("--y-lim", type=float, default=1, help="Output limit")
parser.add_argument("--u-lim", type=float, default=3.5, help="Input limit")
args = parser.parse_args()
set_seed(args.seed)

# load data
wdir, data = read_offline_data(args.wdir)
args_offline, ctrlr = data["args"], data["ctrlr"]
logger = get_logger(os.path.join(wdir, "offline-arx"))

ys, us, true_sys = data["ys"], data["us"], data["true_sys"]
args_offline = data["args"]

with timer() as t:
    arx_sys, _ = estimate_arx_sys(ys, us, order=args.order, max_iter=2500, ignore_Theta=True)
    est_t = t()
    logger.info(f"Time taken to estimate arx sys with GEM: {est_t:.3f}s")
    
J = np.eye(arx_sys.nw * (arx_sys.nx + arx_sys.nu))
th0 = np.zeros(J.shape[1])

Hy = arx_sys.C / args.y_lim
Hu = np.eye(arx_sys.nu) / args.u_lim
H = block_diag(Hy, Hu)
H = np.vstack([H, -H])
init_covar = true_sys.get_steady_post_covar()
init_covar = block_diag(zero_mat(true_sys.nu * (args.order-1)), block_diag(
    true_sys.C @ init_covar @ true_sys.C.T, args.order))

costQ = block_diag(zero_mat(arx_sys.nx - arx_sys.ny), np.eye(arx_sys.ny))
costR = np.eye(arx_sys.ny) * 1e-4
ctrlr = d2pc_pipeline(arx_sys, ys, us, J, th0, args_offline.p_cc, args_offline.delta,
                      costQ, costR, H, init_covar, args.T_err, logger)
with open(os.path.join(wdir, "offline-arx.pkl"), "wb") as f:
    pickle.dump({"sys": arx_sys, "ctrlr": ctrlr, "y_lim": args.y_lim, "u_lim": args.u_lim, "t": est_t}, f)