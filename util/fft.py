import sys
import numpy as np
from scipy.fft import fft2, ifft2
from scipy.optimize import minimize, Bounds

def circshift(mat, amt):
    return np.roll(np.roll(mat, amt[0], axis=0), amt[1], axis=1)

def naive_blur_fft(X, P, center):
    return blur_fft(X, P, center, error = 0)

def blur_fft(X, P, center, error = 0):
    # compute eigenvalues
    S = fft2(circshift(P, (0 - center[0], 0 - center[1])))
    # compute blurred image
    B = np.real(ifft2(np.multiply(S, fft2(X))))
    # normalize to values in range [0, 1]
    B = (B - np.min(B))/(np.max(B) - np.min(B)) # normalizes
    # check if noise is present
    if error == 0:
        return B
    # create noisy matrix with shape of blurred image
    noise = B + (np.random.rand(*B.shape) * error)
    # normalize result again
    return (noise - np.min(noise))/(np.max(noise) - np.min(noise)) # normalizes

def naive_deblur_fft(B, P, center):
    # compute eigenvalues
    S = fft2(circshift(P, (0 - center[0], 0 - center[1])))
    # fix for eigenvalues equal to zero
    # S = np.where(S != 0, S, float('inf'))
    # compute deblurred image
    X = np.real(ifft2(np.divide(fft2(B), S)))
    return X

def gcv(alpha, s, bhat):
    phi_d = np.divide(1, (np.abs(s)**2 + alpha**2))
    G = np.divide(np.sum(np.abs(bhat * phi_d)**2), np.sum(phi_d)**2)
    return G

def gcv_tik(s, bhat):
    # helps alleviate divide by zero issues
    # float_info.epsilon is the smallest possible floating point number greater than zero
    lb = max(np.amin(np.abs(s)), sys.float_info.epsilon)
    ub = np.amax(np.abs(s))
    initial_guess = [1]
    alpha = minimize(gcv, initial_guess, \
                     method='L-BFGS-B', bounds=Bounds(lb, ub), args=(s, bhat), \
                     options={'eps': 1.08})
    return alpha['x'][0]

def tik_deblur_fft(B, P, center, alpha = 0):
    # compute eigenvalues
    S = fft2(circshift(P, (0 - center[0], 0 - center[1])))
    # fix for eigenvalues equal to zero
    S = np.where(S != 0, S, float('inf'))
    s = S.flatten() # should be col-wise
    # calculate regularizaton parameter
    bhat = fft2(B)
    bhat = bhat.flatten()
    # TODO: graph results of varying alphas, compute error between true image and deblurred image
    # alpha = gcv_tik(s, bhat)
    # compute Tikhonov regularized solution
    D = s.conj() * s + abs(alpha)**2
    bhat = s.conj() * bhat
    xhat = np.divide(bhat, D)
    xhat = np.reshape(xhat, B.shape)
    x = np.real(ifft2(xhat))
    return x