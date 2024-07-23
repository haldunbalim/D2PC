
import numpy as np
from d2pc.id_em import get_nll_hess, LTISystem
from d2pc.control import synth_robust_dynof_ctrl, RSMPCController, stoch_tube_design_time_varying, nom_tube_design, get_contraction_rate
from d2pc.utils import *
from typing import Optional
from logging import Logger


def d2pc_pipeline(est_sys: LTISystem, ys:np.ndarray, us:np.ndarray, J: np.ndarray, th0: np.ndarray,
                  p_cc: float, delta: float, costQ: np.ndarray, costR: np.ndarray, H: np.ndarray,
                  init_covar: np.ndarray, T_err: int, logger: Optional[Logger] = None):
    
    """
        Pipeline for the D2PC controller synthesis
        - est_sys: LTISystem - estimated system
        - ys: np.ndarray - output data
        - us: np.ndarray - input data
        - J: np.ndarray - affine parametrization matrix
        - th0: np.ndarray - affine parametrization constant
        - p_cc: float - chance constraint satisfaction prob
        - delta: float - probability that the true system is covered
        - costQ: np.ndarray - cost matrix for the disturbance cov
        - costR: np.ndarray - cost matrix for the measurement noise cov
        - H: np.ndarray - state-input constraints
        - init_covar: np.ndarray - initial covariance of the state
        - T_err: int - error covariance computation horizon
    """
    with timer() as t_off:
        # quantify the parameteric uncertainty
        with timer() as t:
            hess_th = get_nll_hess(est_sys, J, th0, ys, us)
            if logger is not None:
                logger.info("Uncertainty Quantification time: " + f"{t():.3f}"+"s")

        # robust output-feedback controller synthesis
        with timer() as t:
            _, dynof, _, calX = synth_robust_dynof_ctrl(est_sys, J, hess_th, delta, costQ, costR)
            if logger is not None:
                logger.info("Output-feedback controller synthesis time: " +
                            f"{t():.3f}"+"s")

        # offline design of the controller
        with timer() as t:
            ctrlr = RSMPCController(est_sys, dynof, H, p_cc, J, hess_th, delta, costQ, costR)
            # nominal tube design
            try: 
                _, ctrlr.rho, ctrlr.calP = nom_tube_design(est_sys, dynof, H, J, hess_th, delta)
            except:
                if logger is not None:
                    logger.warning("No feasible solution found for nominal tube design, using inv calX from of-controller")
                ctrlr.calP = la.inv(calX)
                ctrlr.rho = get_contraction_rate(est_sys, ctrlr.calP, dynof, J, hess_th, delta)
            if logger is not None:
                logger.info("Nominal tube design time: " + f"{t():.3f}"+"s")
                
        with timer() as t:
            # time-varying stoch tightening
            init_xi_covar = block_diag(init_covar, np.zeros_like(init_covar))
            assert isinstance(T_err, int) and T_err >= 1, "T_err must be a positive integer"
            T_err = 1 if init_xi_covar is None else T_err
            _, sigma_tvs = stoch_tube_design_time_varying(est_sys, dynof, T_err, init_xi_covar, H, J, hess_th, delta)
            ctrlr.sigma_tvs = sigma_tvs
            if logger is not None:
                logger.info("Time-varying stochastic tube design time: " + f"{t():.3f}"+"s")
        ctrlr.compute_constants()
        if logger is not None:
            logger.info("Total offline time: " + f"{t_off():.3f}"+"s")
    return ctrlr