import numpy as np
from d2pc.utils import block_diag
from d2pc.system import LTISystem
from typing import Optional, List
import scipy.linalg as la

class EstimationSetup:
    def __init__(self, init_sys: LTISystem,
                 Qprojectors: Optional[List[np.ndarray]] = None, Qtypes: Optional[List[str]] = None,
                 Rprojectors: Optional[List[np.ndarray]] = None, Rtypes: Optional[List[str]] = None,
                 J: Optional[np.ndarray] = None, th0: Optional[np.ndarray] = None):
        assert isinstance(
            init_sys, LTISystem), "init_sys must be an instance of LTISystem"
        self.sys = init_sys
        nx, nu, ny, nw = init_sys.nx, init_sys.nu, init_sys.ny, init_sys.nw

        Qprojectors = [np.eye(nw)] if Qprojectors is None else Qprojectors
        Qtypes = ["full"] * len(Qprojectors) if Qtypes is None else Qtypes
        self.Qprojectors, self.Qtypes = self.check_projectors(
            Qprojectors, Qtypes, init_sys.Q, "Q", nw)

        Rprojectors = [np.eye(ny)] if Rprojectors is None else Rprojectors
        Rtypes = ["full"] * len(Rprojectors) if Rtypes is None else Rtypes
        self.Rprojectors, self.Rtypes = self.check_projectors(
            Rprojectors, Rtypes, init_sys.R, "R", ny)

        J = np.eye(nw*(nx+nu)) if J is None else J
        th0 = np.zeros(J.shape[0]) if th0 is None else th0
        self.J, self.th0 = self.check_Jth0(J, th0)

    def check_projectors(self, projectors, types, init_mat, _t, dim):
        type(projectors) == list, f"{_t}projectors must be a list"
        assert all(isinstance(p, np.ndarray) for p in projectors), f"{_t}projectors must be a list of numpy arrays"
        projectors = [np.atleast_2d(p) for p in projectors]
        for i, p1 in enumerate(projectors):
            for j, p2 in enumerate(projectors):
                pmul = p1 @ p2.T
                if i == j:
                    assert np.allclose(pmul, np.eye(
                        pmul.shape[0])), f"{_t}projectors must be orthogonal"
                else:
                    assert np.allclose(pmul, np.zeros(
                        pmul.shape)), f"{_t}projectors must be orthogonal"

        assert np.linalg.matrix_rank(block_diag(
            *projectors)) == dim, "Projectors must span the appropriate noise space"
        blocks = []
        rebuilt = np.zeros_like(init_mat)
        for projector in projectors:
            block = projector @ init_mat @ projector.T
            blocks.append(block)
            rebuilt += projector.T @ block @ projector
        assert np.allclose(
            rebuilt, init_mat), f"{_t} matrix of the initial system cannot be decomposed using the provided projectors"

        assert len(projectors) == len(
            types), f"{_t}projectors and {_t}types must have the same length"
        type(types) == list, f"{_t}types must be a list"
        for i, typ in enumerate(types):
            if typ not in ["fixed", "scaled", "full"]:
                raise ValueError(
                    f"{_t}type must be one of 'fixed', 'scaled', 'full', {_t}type[{i}] = {typ}")

        return projectors, types

    def check_Jth0(self, J, th0=None):
        nx, nu, nw = self.sys.nx, self.sys.nu, self.sys.nw
        nTh, nth = J.shape
        assert nTh == nw * \
            (nx + nu), f"J must have nw * (nx + nu) = {nw*(nx+nu)} rows"
        assert nTh >= nth, "J must have at least as many rows as columns"
        assert np.linalg.matrix_rank(J) == nth, "J must have full column rank"

        pvec = (self.sys.E_pinv @ np.hstack([self.sys.A, self.sys.B])).ravel("F")
        assert type(th0) == np.ndarray, "th0 must be a numpy array"
        assert th0.shape[0] == nTh and th0.ndim == 1, "th0 must be a vector of length nTh"
        pvec -= th0

        pvec_re = J @ la.inv(J.T@J) @ J.T @ pvec
        assert np.allclose(pvec, pvec_re), "The given system's process matrices cannot be formed using given J, th0"

        return J, th0