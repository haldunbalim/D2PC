from contextlib import contextmanager
from timeit import default_timer
import numpy as np
import contextlib
import random
import cvxpy as cp
import scipy.linalg as la
from scipy.linalg import lapack
import io
from scipy.sparse import csr_matrix
import logging

def hankelize(signal, window_len):
    signal = signal if signal.ndim == 2 else signal[:, np.newaxis]
    hankel = np.array([signal[i:i+window_len]
                       for i in range(len(signal)-window_len+1)])
    return hankel.reshape(-1, signal.shape[-1]*window_len)

# be careful with this function,
# it assumes that the matrix is triangular with positive diagonals!
def log_det_triangular(triangular_mat):
    return np.log(np.prod(np.diag(triangular_mat)))

def log_det(mat):
    s, logdet = np.linalg.slogdet(mat)
    return s*logdet

def block_diag(*mats):
    if len(mats) == 2 and isinstance(mats[1], int):
        return block_diag(*([mats[0]]*mats[1]))
    else:
        return la.block_diag(*mats)

def block_diag_cvxpy(*mats):
    if len(mats) == 1:
        return mats[0]
    if len(mats) == 2 and isinstance(mats[1], int):
        mat, k = mats
        if k == 1:
            return mat
        h, w = mat.shape
        return cp.vstack([cp.hstack([mat, np.zeros((h, (k-1)*w))]),
                          cp.hstack([np.zeros(((k-1)*h, w)), block_diag_cvxpy(mat, k-1)])])
    elif len(mats) == 2:
        mat1, mat2 = mats
        h1, w1 = mat1.shape
        h2, w2 = mat2.shape
        return cp.vstack([cp.hstack([mat1, np.zeros((h1, w2))]),
                          cp.hstack([np.zeros((h2, w1)), mat2])])
    return block_diag_cvxpy(block_diag_cvxpy(mats[0], mats[1]), *mats[2:])
    
def mat_block_toep(mat, k, shift):
    h, w = mat.shape
    w_ = (k-1)*shift+w
    ret = np.zeros((k*h, w_))
    for i in range(k):
        ret[i*h:(i+1)*h, shift*i:shift*i+w] = mat
    return ret
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def compute_Ms(J, P_smooth):
    return P_smooth[1:] @ np.transpose(J, (0, 2, 1))

def inv_tri(mat, lower=False):
    return la.solve_triangular(mat, np.eye(mat.shape[0]), lower=lower)

def inv_tril(mat):
    return inv_tri(mat, lower=True)

def inv_triu(mat):
    return inv_tri(mat, lower=False)

def chol(mat, lower=False, safe=False):
    try:
        return la.cholesky(mat, lower=lower)
    except Exception as e:
        if safe:
            return la.cholesky(mat + np.eye(mat.shape[0])*2e-8, lower=lower)
        else:
            raise e

def lchol(mat, safe=False):
    return chol(mat, lower=True, safe=safe)

def uchol(mat, safe=False):
    return chol(mat, lower=False, safe=safe)

def rand_psd(n, beta=0):
    A = np.random.rand(n, n) * .5 - .25
    A = A @ A.T 
    if beta != 0:
        A += beta*np.eye(n)
    return A

def symmetrize(x):
    return 0.5 * (x + np.swapaxes(x, -2, -1))

def psd_inverse(m, inds=None, det=False):
    if inds is None:
        inds = np.tri(m.shape[0], k=-1, dtype=bool)
    if m.shape[0] == 1:
        if det:
            return 1./m, np.log(m[0, 0])
        else:
            return 1/m
    # https://stackoverflow.com/questions/40703042/more-efficient-way-to-invert-a-matrix-knowing-it-is-symmetric-and-positive-semi
    cholesky, info = lapack.dpotrf(m)
    if info != 0:
        raise Exception(f"dpotrf failed cond:{np.linalg.cond(m)}")
    inv, info = lapack.dpotri(cholesky)

    if info != 0:
        raise Exception("dpotri failed")
    tri2sym(inv, inds)
    if det:
        logdet = 2 * np.log(np.prod(np.diag(cholesky)))
        return inv, logdet
    else:
        return inv

def tri2sym(m, inds):
    m[inds] = m.T[inds]

def zero_mat(d):
    return np.zeros((d, d))

def dpp_kron(matA, matB):
    h, w = matA.shape
    res = []
    for i in range(h):
        r = []
        for j in range(w):
            r.append(matA[i, j]*matB)
        res.append(r)
    return cp.bmat(res)

def commutation_matrix_sp(m, n):
    row = np.arange(m*n)
    col = row.reshape((m, n), order='F').ravel()
    data = np.ones(m*n, dtype=np.int8)
    K = csr_matrix((data, (row, col)), shape=(m*n, m*n))
    return K.toarray()

def assert_dim_and_pd(mat, dim, name, allow_sd=False):
    assert mat.ndim == 2, f"{name} must be 2D"
    assert mat.shape[0] == mat.shape[1], f"{name} must be square"
    assert mat.shape[0] == dim, f"{name} must be of shape ({dim}, {dim})"
    assert la.ishermitian(mat), f"{name} must be Hermitian"
    min_eigvalsh = np.min(la.eigvalsh(mat))
    if allow_sd:
        assert min_eigvalsh >= 0, f"{name} must be positive semi-definite"
    else:
        assert min_eigvalsh > 0, f"{name} must be positive definite"
    return min_eigvalsh

def sigma_max(mat):
    return np.sqrt(np.max(la.eigvalsh(mat @ mat.T)))

def compute_JDelta(J, nw):
    d = J.shape[0] // nw
    Pcom = commutation_matrix_sp(nw, d)
    Jdelta = np.kron(np.eye(nw), (Pcom @ J).T)
    Jdelta = Jdelta @ np.kron(np.eye(nw).ravel("F")[:, None], np.eye(d))
    return Jdelta

class HiddenPrints(contextlib.redirect_stdout):
    def __init__(self):
        super().__init__(io.StringIO())

@contextmanager
def timer():
    start = default_timer()
    def elapser(): return default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    def elapser(): return end-start


def log_info(msg):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.info(msg)


def log_warning(msg):
    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    logger = logging.getLogger()
    logger.warn(msg)

def setup_ls_arx(order, ys, us):
    U = np.array([la.hankel(_us[:order], _us[order-1:]) for _us in us[1:].T])
    if U.ndim > 2:
        U = np.transpose(U, (1, 0, 2)).reshape(-1, U.shape[-1])
    Y = np.array([la.hankel(_ys[:order], _ys[order-1:])
                    for _ys in ys[:-1].T])
    if Y.ndim > 2:
        Y = np.transpose(Y, (1, 0, 2)).reshape(-1, Y.shape[-1])
    reg = np.concatenate([Y, U]).T
    label = ys[order:]
    return reg, label
