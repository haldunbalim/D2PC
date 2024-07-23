import numpy as np
from d2pc import *
from d2pc.id_em.opt_blocks import OptBlockCF
import unittest
import casadi as ca


class TestOptBlocks(unittest.TestCase):
    def create_stat_matrices(self, nx, nu):
        mat = rand_psd(2*nx+nu, beta=1)
        Sigma = mat[:nx+nu, :nx+nu]
        Psi = mat[nx+nu:, :nx+nu]
        Phi = mat[nx+nu:, nx+nu:]
        return Sigma, Psi, Phi
    
    def test_OptBlockCF_nullJ(self, nx=3, nu=2, nth=7):
        Sigma_b, Psi_b, _ = self.create_stat_matrices(nx, nu)
        Q = rand_psd(nx, nx)

        J = np.zeros((nx*(nx+nu), nth))
        projector = np.eye(nx) 

        opt_block = OptBlockCF(Qtype="fixed", projector=projector, J=J, th0=None)
        AB_new_b, _ = opt_block.optimize(
            AB_curr=None, Q=Q, Sigma=None, Sigma_b=Sigma_b, Psi=None, Psi_b=Psi_b, Phi=None)
        self.assertTrue(np.allclose(AB_new_b, 0))

    def _test_OptBlockCF_subJ(self, J, nx=3, nu=2):
        Sigma, Psi, Phi = self.create_stat_matrices(nx, nu)
        Sigma_b, Psi_b, _ = self.create_stat_matrices(nx, nu)
        projector = np.eye(nx)
        Q = rand_psd(nx, nx)
        Qinv = psd_inverse(Q)

        # solve the problem
        th = cp.Variable(J.shape[1])
        mat = cp.reshape(J@th, (nx, (nx+nu)))
        obj = cp.norm(lchol(Qinv) @ (mat - Psi_b @ psd_inverse(Sigma_b)) @ lchol(Sigma_b), "fro")
        prob = cp.Problem(cp.Minimize(obj))
        prob.solve()
        mat_val = mat.value

        # test the OptBlockCF
        opt_block = OptBlockCF("fixed", projector=projector, J=J, th0=None)
        AB_new, _ = opt_block.optimize(
            AB_curr=None, Q=Q, Sigma=Sigma, Sigma_b=Sigma_b, Psi=Psi, Psi_b=Psi_b, Phi=Phi)
        AB_new = (J @ AB_new).reshape(nx, -1, order="F")
        self.assertTrue(np.allclose(AB_new, mat_val))

    def test_OptBlockCF_dense_subJ(self, nx=3, nu=2):
        J = block_diag(np.eye(nx), np.eye(nx), np.eye(nx))
        J = np.vstack([J, np.zeros((nx, J.shape[1])), np.zeros((nx, J.shape[1]))])
        self._test_OptBlockCF_subJ(J, nx, nu)

    def test_OptBlockCF_sparse_subJ(self, nx=3, nu=2):
        J = block_diag(np.eye(nx), np.eye(nx), np.vstack([np.eye(nx-1), np.zeros((1, nx-1))]))
        J = np.vstack([J, np.zeros((nx, J.shape[1])),np.zeros((nx, J.shape[1]))])
        self._test_OptBlockCF_subJ(J, nx, nu)

    def test_OptBlockCF_fixedQ(self, nx=3, nu=2, nth=7):
        Sigma_b, Psi_b, _ = self.create_stat_matrices(nx, nu)
        Q = rand_psd(nx, nx)

        J = np.zeros((nx*(nx+nu), nth))
        projector = np.eye(nx)

        opt_block = OptBlockCF(
            Qtype="fixed", projector=projector, J=J, th0=None)
        _, Qnew = opt_block.optimize(
            AB_curr=None, Q=Q, Sigma=None, Sigma_b=Sigma_b, Psi=None, Psi_b=Psi_b, Phi=None)
        self.assertTrue(np.allclose(Qnew, Q))

    def _test_OptBlockCF_optQ(self, Qtype, nx=3, nu=2):
        Sigma_b, Psi_b, _ = self.create_stat_matrices(nx, nu)
        Sigma, Psi, Phi = self.create_stat_matrices(nx, nu)
        Q = rand_psd(nx, nx)

        J = np.eye(nx*(nx+nu))
        projector = np.eye(nx)
        th0 = np.random.randn(nx*(nx+nu))

        opt_block = OptBlockCF(
            Qtype=Qtype, projector=projector, J=J, th0=th0)
        AB_new_b, Qnew = opt_block.optimize(
            AB_curr=None, Q=Q, Sigma=Sigma, Sigma_b=Sigma_b, Psi=Psi, Psi_b=Psi_b, Phi=Phi)
        
        AB_new = (AB_new_b + th0).reshape((nx, -1), order="F")

        if Qtype == "full":
            Qinv_cp = cp.Variable((nx, nx), PSD=True)
        elif Qtype == "scaled":
            _lambda = cp.Variable()
            Qinv_cp = _lambda * psd_inverse(Q)
        t = Psi @ AB_new.T
        obj = cp.trace(Qinv_cp @ (Phi + AB_new @ Sigma @ AB_new.T - t - t.T)) - cp.log_det(Qinv_cp)
        prob = cp.Problem(cp.Minimize(obj))
        obj_Q = prob.solve()

        Qnew_inv = psd_inverse(Qnew)
        obj_Qnew = np.trace(Qnew_inv @ (Phi + AB_new @ Sigma @ AB_new.T - t - t.T)) - log_det(Qnew_inv)
        self.assertTrue(obj_Qnew < obj_Q + 1e-8)

    def test_OptBlockCF_scaledQ(self, nx=3, nu=2):
        self._test_OptBlockCF_optQ("scaled", nx, nu)

    def test_OptBlockCF_fullQ(self, nx=3, nu=2):
        self._test_OptBlockCF_optQ("full", nx, nu)

    def test_obj_w_jac(self, nx=6, nu=2, nth=25):
        from d2pc.id_em.lbfgs import obj_w_jac
        J = np.random.rand(nx*(nx+nu), nth)
        th0 = np.random.rand(nx*(nx+nu))
        th = np.random.rand(nth)
        projectors = np.eye(6)
        projectors = [projectors[:2], projectors[2:4], projectors[4:6]]
        Qtypes = ["fixed", "scaled", "full"]
        Q = block_diag(*list(rand_psd(2) for _ in range(3)))
        Q_chol = lchol(Q)
        param_vec = np.concatenate([th, Q_chol.ravel("F")])

        Sigma, Psi, Phi = self.create_stat_matrices(nx, nu)

        # casadi
        param_vec_ca = ca.SX.sym("param_vec", nth+nx**2)
        th_ca = param_vec_ca[:nth]
        Q_chol_vec_ca = param_vec_ca[nth:]

        AB_vec = J @ th_ca + th0
        AB_ca = ca.reshape(AB_vec, (nx, -1))
        Q_chol_ca = ca.reshape(Q_chol_vec_ca, (nx, nx))
        Q_chol_inv_ca = ca.solve(Q_chol_ca, ca.SX.eye(Q_chol_ca.shape[0]))
        Q_inv_ca = Q_chol_inv_ca.T@Q_chol_inv_ca

        t = Psi @ AB_ca.T
        tr = Phi - t - t.T + AB_ca @  Sigma @ AB_ca.T
        obj_ca = ca.trace(Q_inv_ca @ tr) - 2 * \
            ca.sum1(ca.log(ca.fabs(ca.diag(Q_chol_inv_ca))))

        obj_fn = ca.Function("obj_fn", [param_vec_ca], [obj_ca])

        obj, grad = obj_w_jac(param_vec, J, th0, projectors, Qtypes, Sigma, Psi, Phi)
        th_grad = grad[:nth]
        Q_chol_grad = grad[nth:]
        Q_chol_grad = Q_chol_grad.reshape(nx, nx, order="F")
        assert np.allclose(float(obj_fn(param_vec)), obj)

        grad_fn = ca.Function("grad_fn", [param_vec_ca], [ca.gradient(obj_ca, param_vec_ca)])
        grad_ca = grad_fn(param_vec)
        th_grad_ca = np.array(grad_ca[:nth])[:, 0]
        Q_chol_grad_ca = np.array(grad_ca[nth:])[:, 0]
        Q_chol_grad_ca = Q_chol_grad_ca.reshape(nx, nx, order="F")
        Q_chol_grad_ca[np.triu_indices(nx, 1)] = 0

        assert np.allclose(th_grad, th_grad_ca, atol=1e-8)
        for projector, Qtype in zip(projectors, Qtypes):
            Q_chol_grad_sub_ca = projector @ Q_chol_grad_ca @ projector.T
            Q_chol_grad_sub = projector @ Q_chol_grad @ projector.T
            if Qtype == "fixed":
                assert np.allclose(Q_chol_grad_sub, 0)
            elif Qtype == "full":
                assert np.allclose(Q_chol_grad_sub, Q_chol_grad_sub_ca)
            else:
                Q_chol_sub = projector @ Q_chol @ projector.T
                assert np.allclose(
                    np.trace(Q_chol_grad_sub_ca @ Q_chol_sub) * Q_chol_sub, Q_chol_grad_sub)

    

  