import numpy as np
from d2pc import *
import unittest

class TestSpringMass(unittest.TestCase):
    def test_create_spring_mass(self):
        fn = create_spring_mass_sys
        for i in range(3, 6):
            for j in range(2, i+1):
                ms = np.random.rand(i)
                ks = np.random.rand(i)
                ds = np.random.rand(i)
                kwargs = {"num_actuated":j, "ms":ms, "ks":ks, "ds":ds}
                self._test_create_spring_mass(fn, i, **kwargs)

    def _test_create_spring_mass(self, fn, num_masses, **kwargs):
        A, B, C, E, J, th0 = fn(**kwargs)
        AB = np.hstack([A, B])
        vec = spring_mass_vec_fn(**kwargs)

        E_pinv_AB = (J@vec+th0).reshape(num_masses, -1, order="F")
        E_ann = la.null_space(E.T)
        E_ann_pinv = la.pinv(E_ann)
        AB_th = E @ E_pinv_AB + E_ann_pinv.T @ E_ann.T @ AB
        self.assertTrue(np.allclose(AB, AB_th))
    
