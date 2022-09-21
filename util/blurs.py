import math
import random
import numpy as np

def psf_linear(dim: tuple, r: float, theta: float, i: int = None, j: int = None) -> np.ndarray:
    '''
    Returns a point source function blurred with a linear blur. 

    Parameters:
        dim (tuple): dimensions of the image to produce, ex. (3, 3)
        r (float): length of blur in pixels
        theta (float): angle of blur in radians
        i (int, optional): offset for center of psf in dim[0]
        j (int, optional): offset for center of psf in dim[1]

    Returns:
        Normalized matrix for linear blur.
    '''

    # verify that dimensions are greater than 0
    if dim[0] < 1 or dim[1] < 1:
        raise ValueError("Dimensions cannot be less than 1.") 

    # if i and j were not assigned, manually assign them to center of image
    if i is None:
        i = math.ceil(dim[0] / 2)
    if j is None:
        j = math.ceil(dim[0] / 2)

    # verify that offsets are between 0 and dim
    if i < 0:
        i = 0
    if i >= dim[0]:
        i = dim[0] - 1
    if j < 0:
        j = 0
    if j >= dim[1]:
        j = dim[1] - 1



    # create matrix of zeros
    p = np.zeros(dim)

    for radius in range(0, math.floor(r)+1):
        px = math.floor(radius * math.cos(theta) + i) % dim[0]
        py = math.floor(-radius * math.sin(theta) + j) % dim[1]
        p[py, px] = 1

    # normalize p values to [0, 1]
    # p = p / np.linalg.norm(p)

    return p

def blur(func, *args) -> np.ndarray:
    '''
    Allows any arbitrary blur function to be used.
    Arguments should be assigned in order.

    Parameters:
        func (any): which blurring function to use
        args (any): arguments to pass to blurring function
    
    Returns:
        Blurred matrix.
    '''

    return func(*args)

def add_noise(mat: np.ndarray, noise: float) -> np.ndarray:
    '''
    Adds noise to an image.

    Parameters:
        mat (ndarray): matrix to add noise to
        noise (float): amount of noise to add in range [0, 1]

    Returns:
        New matrix with noise added.
    '''
    noisy_mat = mat.copy() + noise * np.random.random(mat.shape)

    # normalize
    norm = np.linalg.norm(noisy_mat)
    # prevent divide by zero errors
    if norm != 0:
        noisy_mat = noisy_mat / np.linalg.norm(noisy_mat)

    return noisy_mat
    

def pad_psf(mat: np.ndarray, size: tuple) -> np.ndarray:
    '''
    Pads a point spread funciton with zeros. The original PSF will
    appear in the upper-lerft corner in order to preserve center
    of PSF for future computations.

    Parameters:
        mat (ndarray): matrix to pad
        size (tuple): desired dimensions of the padded matrix

    Returns:
        Matrix containing the padded data.
    '''

    zeros = np.zeros(size)
    zeros[0:mat.shape[0], 0:mat.shape[1]] = mat
    return zeros