import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import functools


def hadamard(A, B):
    # *
    return np.multiply(A, B)


def khatri_rao(A, B):
    # circle dot
    I = A.shape[0]
    J = A.shape[1]
    K = B.shape[0]
    L = B.shape[1]
    result = np.zeros((I*J, K))
    column_stack = []
    for l in np.arange(L):
        result = kronecker(A[:, l].T.reshape(I, 1), B[:, l].reshape(K, 1))
        column_stack.append(result)
    return np.column_stack(column_stack)


def kronecker(A, B):
    # ⊗
    I = A.shape[0]
    if len(A.shape) == 1:
        J = 1
    else:
        J = A.shape[1]
    K = B.shape[0]
    if len(B.shape) == 1:
        L = 1
    else:
        L = B.shape[1]

    result = np.zeros((I*K, J*L))
    for i in np.arange(I):
        for j in np.arange(J):
            result[i*K:i*K+K, j*L:j*L+L:] = (A[i][j]*B).reshape((K, L))
    return result


def unfold(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')


def cp_als(X, R):
    # init A
    A = np.array([np.random.rand(i, R) for i in X.shape])
    time = 0
    normalize_factor = [1.0]*R
    error = []
    approx_list = []
    while True:
        time += 1
        for n in np.arange(len(X.shape)):
            v_buffer = [np.matmul(A[i].transpose(), A[i])
                        for i in np.arange(len(X.shape)) if i != n]
            V = functools.reduce(hadamard, v_buffer)
            A[n] = functools.reduce(np.matmul, (unfold(X, n), (functools.reduce(
                khatri_rao, [A[i]for i in np.arange(len(A))[::-1] if i != n])), np.linalg.pinv(V)))

            # normalize column
            for i in np.arange(A[n].shape[1]):
                norm = np.linalg.norm(A[n][:, i].transpose())
                A[n][:, i] = A[n][:, i]/norm
                normalize_factor[i] = norm
        # if times==10:
        #     break
        assert (X.shape == cp_compose(normalize_factor, A, R).shape)
        approx = cp_compose(normalize_factor, A, R)
        error.append(np.linalg.norm(approx-X))
        approx_list.append(approx)
        if time >= 20:
            order = np.log(np.abs(np.linalg.norm(error[time-1]-error[time-2]) /
                                  np.linalg.norm(error[time-2]-error[time-3])))/np.log(np.abs(np.linalg.norm(error[time-2]-error[time-3]) / np.linalg.norm(error[time-3]-error[time-4])))
            if np.abs(np.abs(order) - 1) < .01 or order == np.nan or order == np.inf or order == np.negative(np.inf):
                # print(np.linalg.norm(approx-X), ' ', order)
                # print(time)
                break
    return normalize_factor, A, R


def cp_compose(normalize_factor, A, R):
    return functools.reduce(np.add, [functools.reduce(np.multiply.outer, [A[i][:, r]
                                                                          for i in np.arange(len(A))])*normalize_factor[r] for r in np.arange(R)])
