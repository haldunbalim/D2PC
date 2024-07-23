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
logger = get_logger(os.path.join(wdir, "offline-lti"))

ys, us, true_sys = data["ys"], data["us"], data["true_sys"]
args_offline = data["args"]

with timer() as t:
    # setup the estimation problem
    opt_problem = EMRunner(generate_random_lti(
        true_sys.nx, true_sys.ny, true_sys.nu))
    est_sys, _ = opt_problem.em(ys, us, max_iter=20000, rtol=1e-7, verbose=1)
    est_t = t()
    logger.info(f"Time taken to estimate lti sys with EM: {est_t:.3f}s")

with open(os.path.join(wdir, "offline-lti.pkl"), "wb") as f:
    pickle.dump({"sys": est_sys, "t": est_t}, f)
