import numpy as np
import cvxpy as cp
import scipy.linalg as la
from d2pc.utils import *
from d2pc.system import LTISystem
from .dynof_ctrl import DynOFController
from .offline import *
import warnings

class SMPCController:
    def __init__(self, sys: LTISystem, dynof: DynOFController, H: np.ndarray, T_err: float, p_cc: float, 
                 costQ: np.ndarray, costR: np.ndarray):
        """
        MPC controller for the given system, does not consider parametric uncertainty!
            - sys: LTISystem - system
            - dynof: DynOFController - dynamic output-feedback controller
            - H: np.ndarray (nh, nx+nu) - state-input constraints
            - T_err: int - horizon for error covariance bounds to compute
            - p_cc: float - chance constraint level
            - costQ: np.ndarray (nx, nx) - state cost weight
            - costR: np.ndarray (nu, nu) - input cost weight
        """
        self.sys = sys
        self.dynof = dynof

        # state/input constraints
        self.H = H
        self.T_err = T_err
        self.p_cc = p_cc

        # cost matrices
        self.costQ = costQ
        self.costR = costR

    def offline_design(self, init_state_covar: np.ndarray):
        """
        Offline design of the controller.
            - init_state_covar: np.ndarray (nx, nx) - initial state covariance
        """
        # terminal cost
        self.Sc = compute_Sc(self.dynof, self.costQ, self.costR)

        # time-varying stoch tightening
        init_xi_covar = block_diag(
            init_state_covar, np.zeros_like(init_state_covar))
        assert isinstance(
            self.T_err, int) and self.T_err >= 1, "T_err must be a positive integer"
        BdBdT = symmetrize(block_diag(self.sys.E @ self.sys.Q @ self.sys.E.T, self.dynof.L @ self.sys.R @ self.dynof.L.T))
        if self.T_err > 1:
            sigma_tvs = [init_xi_covar]
            for _ in range(self.T_err-2):
                sigma_tvs.append(self.dynof.calA @ sigma_tvs[-1] @ self.dynof.calA.T + BdBdT)
        sigma_tvs.append(ct.dlyap(self.dynof.calA, BdBdT))
        self.sigma_tvs = np.array(sigma_tvs)
        self.stoch_tight_tv = np.array([compute_stoch_tight(sigma_t, self.dynof.K,
                                                            self.H, self.p_cc)
                                        for sigma_t in self.sigma_tvs])
        # compute a scaled ball terminal set that satisfies the assumptions
        self.nom_tight = compute_nom_tight(np.eye(2*self.sys.nx), self.dynof.K, self.H)
        self.underbar_c = compute_underbar_c(self.stoch_tight_tv, self.nom_tight)

    def setup(self, init_state_mean: np.ndarray, T: int, solver: Optional[str] = None):
        """
        Setup the controller for the given system.
            - init_state_mean: np.ndarray (nx,) - initial state mean
            - T: int - time horizon
            - solver: Optional[str] - cvxpy solver to use, default is (MOSEK if approx else ECOS)
        """
        # initialize
        self.T = T
        self.t = 0
        self.dynof.setup()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            init_xi_mean = np.hstack(
                [init_state_mean, np.zeros(self.sys.nx)])
            self.prob = self.construct_control_prob(
                init_xi_mean, solver)

        # to store the results
        self.nus = []
        self.objs = []

    def _construct_control_prob(self):
        """
        Construct the optimization problem.
            - init_state_mean: np.ndarray (nx,) - initial state mean
            - use_tv_tightening: bool - whether to use time-varying tightening
        """
        sys = self.sys
        nx, nu, nw = sys.nx, sys.nu, sys.nw

        # params
        init_xi = cp.Parameter(nx * 2)
        stoch_tight = cp.Parameter((self.T, self.H.shape[0]))
        # vars
        xis = cp.Variable((self.T+1, nx*2))
        input_terms = cp.Variable((self.T, nu))  # \nu
        # matrices
        calA = self.dynof.calA
        calBv = self.dynof.calBv
        IK = block_diag(np.eye(nx), self.dynof.K)
        # compute matrix sqrt for computing norms
        Qc_sqrt = la.sqrtm(self.costQ)
        Rc_sqrt = la.sqrtm(self.costR)
        costXi_sqrt = block_diag(Qc_sqrt, Rc_sqrt)
        Sc_sqrt = la.sqrtm(self.Sc)

        # initial condition:
        constraints = [xis[0] == init_xi]
        cost = 0  # initialize cost
        for i in range(self.T):
            xu = IK @ xis[i] + cp.hstack([np.zeros(nx), input_terms[i]])
            # state / input constraints:
            h = np.ones(self.H.shape[0]) - stoch_tight[i] 
            constraints.append(self.H @ xu <= h)

            # stage cost:
            cost += cp.sum_squares(costXi_sqrt @ xu)  # state cost

            # nom prediction dyn:
            constraints.append(xis[i+1] == calA @ xis[i] + calBv @ input_terms[i])

        # terminal cost:
        cost += cp.sum_squares(Sc_sqrt @ xis[-1])

        # terminal constraint:
        constraints += [cp.norm(xis[-1]) <= self.underbar_c]

        # form control problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        return prob, (init_xi, stoch_tight), (xis, input_terms)

    def construct_control_prob(self, init_xi_mean: np.ndarray, solver: Optional[str] = None):
        """
        Construct the function that calls the optimization problem.
            - init_state_mean: np.ndarray (nx,) - initial state mean
            - solver: Optional[str] - cvxpy solver to use
        """
        # construct the optimization problem
        prob, params, vars = self._construct_control_prob()
        init_xi, stoch_tight = params
        xis, input_terms = vars

        # store relevant variables
        nxt_xi = xis[1]
        inp = input_terms[0]

        if solver is None:
            solver = cp.ECOS

        # solve one time with dummy parameter values for caching the solver
        init_xi.value = np.zeros(init_xi.shape)
        stoch_tight.value = np.zeros(stoch_tight.shape)
        prob.solve(solver=solver)

        def solve():
            # stoch tightening
            if self.t <= self.T_err:
                tv = self.stoch_tight_tv[self.t:min(
                    len(self.stoch_tight_tv), self.t+self.T)]
                if len(tv) < self.T:
                    inf = np.repeat(
                        self.stoch_tight_tv[-1][None, :], self.T-len(tv), axis=0)
                    tv = np.vstack([tv, inf])
                stoch_tight.value = tv
            # inital values
            if self.t == 0:
                init_xi.value = init_xi_mean
            else:
                # shift the dynamics (1|t-1 -> 0|t)
                init_xi.value = nxt_xi.value

            # solve the control problem
            try:
                obj = prob.solve(solver=solver, warm_start=True)
            except cp.error.SolverError:
                raise Exception(f"Solver error at time {len(self.nus)}")
            if prob.status != cp.OPTIMAL:
                if self.t == 0:
                    raise Exception(f"Infeasible problem at initial state")
                else:
                    # this should never happen
                    raise Exception(
                        f"Recursive feasibility failed at time {len(self.nus)}")
            # store the results
            self.objs.append(obj)
            self.nus.append(inp.value)
            return inp.value
        return solve

    def control(self, y: np.ndarray):
        """
            Compute the control input.
            y: np.ndarray (ny,) - measurement
        """
        # compute controller input
        nu = self.prob()
        # get the dynamic output feedback control
        u_dynof = self.dynof.control(y)
        self.t += 1  # increment time
        return u_dynof + nu  # return applied input

    def compute_input_sequence(self, Tsim):
        for _ in range(Tsim):
            self.prob()
            self.t += 1
        return np.array(self.nus)
