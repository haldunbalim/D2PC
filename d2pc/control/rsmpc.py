import numpy as np
import cvxpy as cp
import scipy.linalg as la
from d2pc.utils import *
from d2pc.system import LTISystem
from .dynof_ctrl import DynOFController
from .offline import *
from typing import Optional
import warnings

class RSMPCController:
    def __init__(self, sys: LTISystem, dynof: DynOFController, H: np.ndarray, p_cc: float, J: np.ndarray,
                 hess_th: np.ndarray, delta: float, costQ: np.ndarray, costR: np.ndarray):
        """
        Proposed Robust SMPC controller for the given uncertain system.
            - sys: LTISystem - system
            - dynof: DynOFController - dynamic output-feedback controller
            - H: np.ndarray (nh, nx+nu) - state-input constraints
            - p_cc: float - chance constraint level
            - J: np.ndarray (nw*(nx+nu), nth) - affine parametrization matrix
            - hess_th: np.ndarray (nth, nth) - Hessian of the uncertainty
            - delta: float - confidence level for parametric uncertainty
            - costQ: np.ndarray (nx, nx) - state cost weight
            - costR: np.ndarray (nu, nu) - input cost weight
        """
        self.sys = sys
        self.dynof = dynof

        # state/input constraints
        self.H = H
        self.p_cc = p_cc

        # param unc
        self.J = J
        self.hess_th = hess_th
        self.covar_th = la.inv(hess_th)
        self.delta = delta

        # cost matrices
        self.costQ = costQ
        self.costR = costR

        # initialize
        self.calP = None
        self.rho = None
        self.sigma_tvs = None

    def compute_constants(self):
        """
        Compute offline constants of the controller.
        """
        # constraint tightening
        if self.calP is None:
            raise Exception("Nominal tube design is not set")
        if self.rho is None:
            raise Exception("Contraction rate is not set")
        self.nom_tight = compute_nom_tight(self.calP, self.dynof.K, self.H)
        if self.sigma_tvs is None:
            raise Exception("Stochastic tube design is not set")
        self.stoch_tight_tv = np.array([compute_stoch_tight(sigma_t, self.dynof.K,
                                                            self.H, self.p_cc)
                                        for sigma_t in self.sigma_tvs])

        # terminal cost
        self.Sc = compute_Sc(self.dynof, self.costQ, self.costR)

        # terminal set
        self.bar_sigma = compute_bar_sigma(
            self.sys, self.J, self.covar_th, self.calP, self.dynof.K)
        self.underbar_c = compute_underbar_c(
            self.stoch_tight_tv, self.nom_tight)

    def setup(self, init_state_mean:np.ndarray, T: int, approx: Optional[bool] = False, 
              solver: Optional[str] = None, **solver_args: dict):
        """
        Setup the controller for the given system.
            - init_state_mean: np.ndarray (nx,) - initial state mean
            - T: int - time horizon
            - approx: bool - whether to use the approximate tube dynamics
            - solver: Optional[str] - cvxpy solver to use, default is (MOSEK if approx else ECOS)
        """
        # initialize
        self.T = T
        self.t = 0
        self.dynof.setup()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            init_xi_mean = np.hstack([init_state_mean, np.zeros(self.sys.nx)])
            self.prob = self.construct_control_prob(init_xi_mean, approx, solver, **solver_args)

        # to store the results
        self.nus = []
        self.alphas = []
        self.xis = []
        self.objs = []

    def _construct_control_prob(self, approx: bool):
        """
        Construct the optimization problem.
            - approx: bool - whether to use the approximate tube dynamics
        """
        sys = self.sys
        nx, nu, nw = sys.nx, sys.nu, sys.nw
        # prob scaling constant for param uncertainty
        d_sqrt = np.sqrt(chi2.ppf(self.delta, self.J.shape[1]))

        # params
        init_xi = cp.Parameter(nx * 2)
        init_alpha = cp.Parameter(nonneg=True)
        stoch_tight = cp.Parameter((self.T, self.H.shape[0]))
        # vars
        xis = cp.Variable((self.T+1, nx*2))
        alphas = cp.Variable((self.T+1))
        input_terms = cp.Variable((self.T, nu)) # \nu
        # matrices
        calA = self.dynof.calA
        calBv = self.dynof.calBv
        IK = block_diag(np.eye(nx), self.dynof.K)
        # compute matrix sqrt for computing norms
        calP_sqrt = la.sqrtm(self.calP)
        covar_th_sqrt = la.sqrtm(self.covar_th)
        Sc_sqrt = la.sqrtm(self.Sc)
        Qc_sqrt = la.sqrtm(self.costQ)
        Rc_sqrt = la.sqrtm(self.costR)
        Xic_sqrt = block_diag(Qc_sqrt, Rc_sqrt)

        # required matrices for tube dynamics
        Bp = np.vstack([sys.E, np.zeros((nx, nw))])
        PsqrtBp = calP_sqrt @ Bp
        Sigma_J_sqrt = np.kron(np.eye(nx+nu), PsqrtBp) @ self.J @ covar_th_sqrt
        if approx:
            # approximate tube dynamics
            Sigma_J = Sigma_J_sqrt @ Sigma_J_sqrt.T
            I = np.eye(2*nx)
            Ikes = [np.kron(np.eye(nx+nu), I[i][:, None]) for i in range(2*nx)]
            Sigma_bar = np.sum([Ike.T @ Sigma_J @ Ike for Ike in Ikes], axis=0)
            Sigma_bar_sqrt = la.sqrtm(Sigma_bar)

        # initial condition:
        constraints = [xis[0] == init_xi, alphas[0] == init_alpha]
        cost = 0  # initialize cost
        for i in range(self.T):
            _xu = IK @ xis[i]
            xu = _xu + cp.hstack([np.zeros(nx), input_terms[i]])
            # stage cost:
            cost += cp.sum_squares(Xic_sqrt @ _xu) + cp.sum_squares(Rc_sqrt @ input_terms[i])
            # state / input constraints:
            tightening = stoch_tight[i] + self.nom_tight * alphas[i]
            constraints.append(
                self.H @ xu <= np.ones(self.H.shape[0]) - tightening)

            # tube dynamics offset:
            if approx:
                offset = cp.norm(Sigma_bar_sqrt @ xu)
            else:
                offset = cp.sigma_max(
                    dpp_kron(xu[None, :], np.eye(2*nx)) @ Sigma_J_sqrt)

            # tube dynamics:
            constraints.append(
                alphas[i+1] >= self.rho * alphas[i] + d_sqrt * offset)

            # nom prediction dyn:
            constraints.append(xis[i+1] == calA @ xis[i] + calBv @ input_terms[i])

        # terminal cost:
        cost += cp.sum_squares(Sc_sqrt @ xis[-1])

        # terminal constraint:
        constraints += [cp.norm(calP_sqrt@xis[-1]) +
                        alphas[-1] <= self.underbar_c]
        if self.bar_sigma >= 1-self.rho:
            constraints += [cp.norm(calP_sqrt @ xis[-1]) <=
                            (1-self.rho) * self.underbar_c / self.bar_sigma]
        # form control problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        return prob, (init_xi, init_alpha, stoch_tight), (xis, alphas, input_terms)

    def construct_control_prob(self, init_xi_mean: np.ndarray, approx: bool, solver: Optional[str] = None, **solver_args: dict):
        """
        Construct the function that calls the optimization problem.
            - init_state_mean: np.ndarray (nx,) - initial state mean
            - approx: bool - whether to use the approximate tube dynamics
            - solver: Optional[str] - cvxpy solver to use
        """
        # construct the optimization problem
        prob, params, vars = self._construct_control_prob(approx)
        init_xi, init_alpha, stoch_tight = params
        xis, alphas, input_terms = vars

        # store relevant variables
        nxt_xi, nxt_alpha = xis[1], alphas[1]
        inp = input_terms[0]

        if solver is None:
            solver = cp.ECOS if approx else cp.MOSEK

        # solve one time with dummy parameter values for caching the solver
        init_xi.value = np.zeros(init_xi.shape)
        init_alpha.value = 0
        stoch_tight.value = np.zeros(stoch_tight.shape)
        prob.solve(solver=solver, **solver_args)

        def solve():
            # stoch tightening
            if self.t <= len(self.stoch_tight_tv):
                tv = self.stoch_tight_tv[self.t:min(len(self.stoch_tight_tv), self.t+self.T)]
                if len(tv) < self.T:
                    inf = np.repeat(self.stoch_tight_tv[-1][None, :], self.T-len(tv), axis=0)
                    tv = np.vstack([tv, inf])
                stoch_tight.value = tv
            # inital values
            if self.t == 0:
                init_xi.value = init_xi_mean
                init_alpha.value = 0
            else:
                # shift the tube dynamics (1|t-1 -> 0|t)
                init_xi.value = nxt_xi.value
                init_alpha.value = nxt_alpha.value

            # solve the control problem
            try:
                obj = prob.solve(solver=solver, **solver_args)
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
            self.alphas.append(alphas[0].value)
            self.xis.append(xis[0].value)
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
