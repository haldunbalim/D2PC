import numpy as np
from d2pc import *
from d2pc.id_em.unc_quant import apply_kf_csadi
import casadi as ca
import unittest


class TestUncQuant(unittest.TestCase):
    def test_apply_kf_csadi(self, nx=3, ny=2, nu=2):
        sys = generate_random_lti(nx, ny, nu)

        _, ys, us = sys.simulate(200, RandomNormalController(nu))
        true_ll = np.sum(sys.kf_fwd(ys, us)[-1])

        pvec_cs = ca.SX.sym("pvec", nx*(nx+nu))
        pvec = np.hstack([sys.A, sys.B]).ravel("F")
        AB_cs = pvec_cs.reshape((nx, nx+nu))
        
        ll_tot = apply_kf_csadi(AB_cs[:, :nx], AB_cs[:, nx:], sys.C, sys.E, sys.Q,
                                sys.R, sys.mu0, sys.P0, ys, us, steady_idx=-1)
        ll_tot = ca.Function("f", [pvec_cs], [ll_tot])(pvec)
        self.assertTrue(np.allclose(ll_tot, true_ll), "KF not working")
        
        steady_idx = sys.get_kf_fwd_steady_idx()
        ll_tot_st = apply_kf_csadi(AB_cs[:, :nx], AB_cs[:, nx:], sys.C, sys.E, sys.Q,
                                sys.R, sys.mu0, sys.P0, ys, us, steady_idx=steady_idx)
        ll_tot_st = ca.Function("f", [pvec_cs], [ll_tot_st])(pvec)
        self.assertTrue(np.allclose(ll_tot_st, true_ll), "Steady state filter not working")
