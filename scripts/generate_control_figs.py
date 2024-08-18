import argparse
import os
import numpy as np
from d2pc import *
from d2pc.experiment.figure import *

# args for true system, and estimation problem
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="Random Seed")
parser.add_argument("--wdir", type=str, default=None,
                    help="Folder name for the current experiment, if not provided retrieves last one in outputs folder")
parser.add_argument("-T", type=int, default=30, help="Control horizon")
parser.add_argument("--T-err-max", type=int, default=30,
                    help="Max plotting horizon for error covariance bound")
parser.add_argument("-T-sim", type=int, default=100,
                    help="Control simulation length")
parser.add_argument("--N-sim", type=int, default=100000,
                    help="number of simulations for the control")
parser.add_argument("--mass-nr", type=int, default=4, help="Mass number to plot")
parser.add_argument("--recompute", action="store_true",
                    help="Recompute the results")
args = parser.parse_args()
print(type(args.seed))
set_seed(args.seed)

# load data
wdir, data = read_offline_data(args.wdir)
args_offline, ctrlr = data["args"], data["ctrlr"]
logger = get_logger(os.path.join(wdir, "control"))

if not os.path.exists(os.path.join(wdir, "control.pkl")) or args.recompute:
    true_sys, est_sys = data["true_sys"], data["est_sys"]
    diag = [1, 0] * (true_sys.nx // 2)
    init_mean = np.array(diag) * -.5
    init_covar = data["init_covar"]
    
    # compute input sequence computed by the controller using the tube dynamics
    ctrlr.setup(init_mean, args.T, approx=False)
    with timer() as t:
        nus_lmi = ctrlr.compute_input_sequence(args.T_sim)
        t_avg_lmi = t() / args.T_sim

    # compute input sequence computed by the controller using the approximate tube dynamics
    ctrlr.setup(init_mean, args.T, approx=True)
    with timer() as t:
        nus_soc = ctrlr.compute_input_sequence(args.T_sim)
        t_avg_soc = t() / args.T_sim

    est_dynof = synth_dynof_ctrl(est_sys, ctrlr.costQ, ctrlr.costR)
    smpc_ctrlr = SMPCController(
        est_sys, est_dynof, ctrlr.H, args_offline.T_err, ctrlr.p_cc, ctrlr.costQ, ctrlr.costR)
    smpc_ctrlr.offline_design(init_covar)
    smpc_ctrlr.setup(init_mean, args.T)
    with timer() as t:
        nus_smpc = smpc_ctrlr.compute_input_sequence(args.T_sim)
        t_avg_smpc = t() / args.T_sim

    # compute data for the control sequence
    xs_lmi, _, _, costs_lmi = simulate_ctrlr_multi(args.N_sim, true_sys, ctrlr.dynof, nus_lmi,
                                                   init_mean, init_covar, ctrlr.costQ, ctrlr.costR)
    (xs_mean_lmi, xs_std_lmi, xs_vio_mean_lmi,
     max_vio_lmi, max_vio_time_lmi,
     max_vio_channel_lmi, avg_cost_lmi) = process_ctrlr_data(xs_lmi, costs_lmi, args_offline.v_lim)

    xs_soc, _, _, costs_soc = simulate_ctrlr_multi(args.N_sim, true_sys, ctrlr.dynof, nus_soc,
                                                   init_mean, init_covar, ctrlr.costQ, ctrlr.costR)
    (xs_mean_soc, xs_std_soc, xs_vio_mean_soc,
     max_vio_soc, max_vio_time_soc,
     max_vio_channel_soc, avg_cost_soc) = process_ctrlr_data(xs_soc, costs_soc, args_offline.v_lim)

    xs_smpc, ys_smpc, us_smpc, costs_smpc = simulate_ctrlr_multi(args.N_sim, true_sys, est_dynof, nus_smpc,
                                                                 init_mean, init_covar, ctrlr.costQ, ctrlr.costR)
    (xs_mean_smpc, xs_std_smpc, xs_vio_mean_smpc,
     max_vio_smpc, max_vio_time_smpc,
     max_vio_channel_smpc, avg_cost_smpc) = process_ctrlr_data(xs_smpc, costs_smpc, args_offline.v_lim)

    xs_dynof, ys_dynof, us_dynof, costs_dynof = simulate_ctrlr_multi(args.N_sim, true_sys, ctrlr.dynof, np.zeros_like(nus_lmi),
                                                                     init_mean, init_covar, ctrlr.costQ, ctrlr.costR)
    (xs_mean_dynof, xs_std_dynof, xs_vio_mean_dynof,
     max_vio_dynof, max_vio_time_dynof,
     max_vio_channel_dynof, avg_cost_dynof) = process_ctrlr_data(xs_dynof, costs_dynof, args_offline.v_lim)

    # compute tubes for the nominal term
    tubes_lmi, tubes_soc, _ = fwd_reach_nom_tube(ctrlr, nus_lmi, init_mean)
    tubes_sampling = fwd_reach_nom_tube_sampling(ctrlr, nus_lmi, init_mean)

    # compute tubes for the error term
    stoch_tights_sampling, stoch_tights_mean = compute_constr_tight_sampling(
        ctrlr, est_sys, init_covar, T_err_plot=args.T_err_max)
    with open(os.path.join(wdir, "control.pkl"), "wb") as f:
        pickle.dump({ "xs_mean_lmi": xs_mean_lmi, "xs_std_lmi": xs_std_lmi, "xs_vio_mean_lmi": xs_vio_mean_lmi,
                      "xs_mean_soc": xs_mean_soc, "xs_std_soc": xs_std_soc, "xs_vio_mean_soc": xs_vio_mean_soc,
                      "xs_mean_smpc": xs_mean_smpc, "xs_std_smpc": xs_std_smpc, "xs_vio_mean_smpc": xs_vio_mean_smpc,
                      "xs_mean_dynof": xs_mean_dynof, "xs_std_dynof": xs_std_dynof, "xs_vio_mean_dynof": xs_vio_mean_dynof,
                      "tubes_lmi": tubes_lmi, "tubes_soc": tubes_soc, "tubes_sampling": tubes_sampling,
                      "stoch_tights_sampling": stoch_tights_sampling, "stoch_tights_mean": stoch_tights_mean,
                      "t_avg_lmi": t_avg_lmi, "t_avg_soc": t_avg_soc, "t_avg_smpc": t_avg_smpc,
                      "nus_lmi": nus_lmi, "nus_soc": nus_soc, "nus_smpc": nus_smpc, "nus_dynof": np.zeros_like(nus_lmi),
                      "avg_cost_lmi": avg_cost_lmi, "avg_cost_soc": avg_cost_soc, "avg_cost_smpc": avg_cost_smpc, "avg_cost_dynof": avg_cost_dynof,
                      "max_vio_lmi": max_vio_lmi, "max_vio_time_lmi": max_vio_time_lmi, "max_vio_channel_lmi": max_vio_channel_lmi,
                      "max_vio_soc": max_vio_soc, "max_vio_time_soc": max_vio_time_soc, "max_vio_channel_soc": max_vio_channel_soc,
                      "max_vio_smpc": max_vio_smpc, "max_vio_time_smpc": max_vio_time_smpc, "max_vio_channel_smpc": max_vio_channel_smpc,                      
                      "max_vio_dynof": max_vio_dynof, "max_vio_time_dynof": max_vio_time_dynof, "max_vio_channel_dynof": max_vio_channel_dynof}, f)
else:
    with open(os.path.join(wdir, "control.pkl"), "rb") as f:
        data = pickle.load(f)
        (xs_mean_lmi, xs_std_lmi, xs_vio_mean_lmi,
         xs_mean_soc, xs_std_soc, xs_vio_mean_soc,
         xs_mean_smpc, xs_std_smpc, xs_vio_mean_smpc,
         xs_mean_dynof, xs_std_dynof, xs_vio_mean_dynof,
         tubes_lmi, tubes_soc, tubes_sampling,
         stoch_tights_sampling, stoch_tights_mean,
         t_avg_lmi, t_avg_soc, t_avg_smpc,
         nus_lmi, nus_soc, nus_smpc, nus_dynof,
         avg_cost_lmi, avg_cost_soc, avg_cost_smpc, avg_cost_dynof,
         max_vio_lmi, max_vio_time_lmi, max_vio_channel_lmi,
         max_vio_soc, max_vio_time_soc, max_vio_channel_soc,
         max_vio_smpc, max_vio_time_smpc, max_vio_channel_smpc,
         max_vio_dynof, max_vio_time_dynof, max_vio_channel_dynof) = data.values()

# log the results
logger.info(
    f"Average time per action for LMI-based controller: {t_avg_lmi:.3f}"+"s")
logger.info(
    f"Average time per action for SOC-based controller: {t_avg_soc:.3f}"+"s")
logger.info(
    f"Average time per action for Nominal MPC controller: {t_avg_smpc:.3f}"+"s")
logger.info(" ")

logger.info(f"Normalized average control cost for LMI-based controller: {(avg_cost_lmi / avg_cost_dynof):.3f}")
logger.info(f"Normalized average control cost for SOC-based controller: {(avg_cost_soc / avg_cost_dynof):.3f}")
logger.info(
    f"Normalized average control cost for Nominal MPC controller: {(avg_cost_smpc / avg_cost_dynof):.3f}")
logger.info(
    f"Normalized average control cost for Dynamic OF controller: {(avg_cost_dynof / avg_cost_dynof):.3f}")
logger.info(" ")

logger.info(
    f"Maximum constraint violation probability for LMI-based controller: {max_vio_lmi:.4f} @ t={max_vio_time_lmi}")
logger.info(
    f"Maximum constraint violation probability for SOC-based controller: {max_vio_soc:.4f} @ t={max_vio_time_soc}")
logger.info(
    f"Maximum constraint violation probability for SMPC controller: {max_vio_smpc:.4f} @ t={max_vio_time_smpc}")

set_fig_params()
# plot the control sequence
fig = control_plot(xs_mean_lmi, xs_std_lmi, xs_vio_mean_lmi, xs_mean_soc, xs_std_soc, xs_vio_mean_soc,
                   args_offline.v_lim, args_offline.p_cc, mass_nr=args.mass_nr)
fig.savefig(os.path.join(wdir, "control.png"))

# plot the fwd reach for the nominal tubes
fig = nom_tube_plot(tubes_lmi, tubes_soc, tubes_sampling)
fig.savefig(os.path.join(wdir, "nom-tube.png"))

# plot the constraint tightening
fig = plot_constraint_tightening(ctrlr.stoch_tight_tv, stoch_tights_sampling, stoch_tights_mean, mass_nr=args.mass_nr)
fig.savefig(os.path.join(wdir, "constr-tightening.png"))


