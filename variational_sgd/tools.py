import numpy as np


def lyapunov(A, B, H, eps):
    """ Returns the solution of the Lyapunov equation given HA sdp matrix """
    U, S, _ = np.linalg.svd(H @ A)
    sigma = np.zeros(A.shape)
    for i in range(len(A)):
        for j in range(len(A)):
            sigma += (1 / (S[i] + S[j])) * U[:, i][np.newaxis].T @ U.T[i][np.newaxis] @ H @ B @ B.T @ H.T @ U[:, j][
                np.newaxis].T @ U.T[j][np.newaxis]
    return eps * sigma
