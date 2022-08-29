import math
import numpy as np

def psf_gaussian(dim: tuple, s1: float = 0, s2: float = 0, i: int = 0, j: int = 0) -> np.ndarray:
    '''
    Returns a point spread function using a Gaussian blur.

    Parameters:
        dim (tuple): dimensions of the image to produce, ex. (3,3) for a 3x3 kernel
        s1 (float): scaling factor for dim[0]
        s2 (float): scaling factor for dim[1]
        i (int): offset for center of psf in dim[0]
        j (int): offset for center of psf in dim[1]

    Returns:
        Normalized matrix for Gaussian blur.
    '''

    # verify that kern size is not zero or negative
    if dim[0] < 1 or dim[1] < 1:
        raise ValueError("Dimensions must be greater than or equal to 1.")

    # verify that i and j offsets are in range [0, dim - 1]
    if i < 0 or i >= dim[0] - 1:
        raise ValueError("i must be in range of [0, dim - 1]")
    if j < 0 or j >= dim[1] - 1:
        raise ValueError("j must be in range of [0, dim - 1]")

    # create matrix of zeros
    p = np.zeros(dim)

    # apply blur
    for x in range(0, dim[0]):
        for y in range(0, dim[1]):
            p[x, y] = math.exp(-0.5 * ((x - i)/s1)**2 - 0.5 * ((y - j)/s2)**2)

    # normalize p values to [0, 1]
    p = p / np.linalg.norm(p)

    return p

if __name__ == '__main__':
    psf_gaussian((16, 16), 0.2, 0.2, 7, 7)