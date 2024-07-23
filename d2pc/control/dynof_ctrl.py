import numpy as np
import scipy.linalg as la
from d2pc.utils import *
import cvxpy as cp
from scipy.stats import chi2
from d2pc.system import LTISystem
from functools import partial


class DynOFController:
    def __init__(self, sys: LTISystem, Ac: np.ndarray, K: np.ndarray, L: np.ndarray):
        self.sys = sys
        self.Ac = Ac
        self.K = K
        self.L = L

    def setup(self):
        self.xc = np.zeros(self.sys.nx)

    def control(self, y: np.ndarray):
        """
            Compute control input
            - y (np.ndarray) (ny): measurement
        """
        # compute input
        u = self.K @ self.xc
        # update controller state
        self.xc = self.Ac @ self.xc + self.L @ y
        return u
    
    @property
    def calA(self):
        return np.block([[self.sys.A, self.sys.B @ self.K], [self.L @ self.sys.C, self.Ac]])
    
    @property
    def calBv(self):
        return np.block([[self.sys.B], [np.zeros((self.sys.nx, self.sys.nu))]])
