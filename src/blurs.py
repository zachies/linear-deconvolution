import math
import numpy as np

from scipy.fft import fft2, ifft2, fftshift

def gaussian(dimensions: tuple, s1: float = 0, s2: float = 0, **kwargs) -> np.ndarray:
    '''
    Returns a point spread function of a Gaussian form with 
    the supplied dimensions and spread factors.

    Parameters:
        dimensions (tuple): dimensions of the image to produce, ex. (3,3) for a 3x3 kernel
        s1 (float): scaling factor for dimensions[0]
        s2 (float): scaling factor for dimensions[1]
        kwargs (dict): additional parameters

    Returns:
        Normalized real components of the point spread function in FFT space.
    '''

    i = math.floor(dimensions[0] / 2)
    j = math.floor(dimensions[1] / 2)

    mat = np.zeros(dimensions)
    for x in range(0, dimensions[0]):
        for y in range(0, dimensions[1]):
            mat[x, y] = math.exp(-0.5 * ((x - i)/s1)**2 - 0.5 * ((y - j)/s2)**2)
    mat = mat / np.sum(mat)
    return np.real(fft2(fftshift(mat)))

    # n = dimensions[0]
    # width = s1
    # x,y = np.mgrid[:n,:n]
    # g = (x-n/2)**2 + (y-n/2)**2 < width**2
    # g = g/np.sum(g)
    # return fft2(fftshift(g)).real

def linear(dimensions: tuple, radius: float = 0, angle: float = 0, **kwargs) -> np.ndarray:
    '''
    Returns a point spread function of linear form with 
    the supplied dimensions and spread factors. Equivalent to linear blur.

    Parameters:
        dimensions (tuple): dimensions of the image to produce, ex. (3,3) for a 3x3 kernel
        radius (float): length of linear blur
        angle (float): angle of linear blur
        kwargs (dict): additional parameters

    Returns:
        Normalized real components of the point spread function in FFT space.
    '''

    p = np.zeros(dimensions)

    # get center of psf
    i = math.ceil(p.shape[0] / 2)
    j = math.ceil(p.shape[1] / 2)

    for r in range(0, math.floor(radius)):
        px = math.floor(r * math.cos(angle) + i) % p.shape[0]
        py = math.floor(r * math.sin(angle) + j) % p.shape[1]
        p[py, px] = 1
    
    p = p / np.sum(p)

    return fft2(fftshift(p))

def radial(dim: tuple, radius: float) -> np.ndarray:
    '''
    Returns a point source function blurred with a radial blur. 
    Parameters:
        dim (tuple): dimensions of the image to produce, ex. (3, 3)
        radius (float): radius of blur in pixels
    Returns:
        Normalized matrix for radial blur.
    '''

    # create matrix of zeros
    p = np.zeros(dim)

    # set intensity at center of image to 1
    i = math.ceil(p.shape[0] / 2)
    j = math.ceil(p.shape[1] / 2)
    p[i, j] = 1

    # use Moffat blur where rho = 0
    for x in range(0, dim[0]):
        for y in range(0, dim[1]):
            if (x - i)**2 + (y - j)**2 <= radius**2:
                p[x, y] = 1

    return fft2(fftshift(p))

def noise(mat: np.ndarray, noise: float) -> np.ndarray:
    '''
    Adds noise to a matrix. The noise follows a standard distribution.

    Parameters:
        mat (ndarray): matrix to add noise to
        noise (float): scaling factor for noise, should be in range [0, 1]
    
    Returns:
        Matrix with added noise.
    '''

    return mat + noise * np.random.randn(*mat.shape)

def blur(mat: np.ndarray, filter: np.ndarray) -> np.ndarray:
    '''
    Blurs a matrix with the given filter using FFT.

    Parameters:
        mat (ndarray): matrix representing original image
        filter (ndarray): matrix representing blur function

    Returns:
        Blurred matrix.
    '''
    return np.real(ifft2(filter * fft2(mat)))