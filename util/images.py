import numpy as np
from PIL import Image
from util.borders import undo

def im_float_to_bmp(mat):
    '''
        Converts floating-point values in a matrix to [0, 255].
        Useful for saving the image.
    '''
    return (mat * 255 / np.max(mat)).astype('uint8')

def load_img(path):
    '''
        Returns a matrix from a given image
    '''
    mat = np.asarray(Image.open(path))
    return mat / 255

def save_img(mat, path, kern_size = None):
    '''
        Saves an image to the specified path. If a kernel size is given,
        it will undo the boundary conditions before saving. Otherwise, it will unpack
        it assuming a full 3x3 deconstruction.
    '''
    Image.fromarray(im_float_to_bmp(undo(mat, kern_size))).save(path)

def save_img_raw(mat, path):
    '''
        Saves an image to the specified path. Assumes no undoing of boundary conditions.
    '''
    Image.fromarray(im_float_to_bmp(mat)).save(path)

def pad_psf(mat, shape):
    '''
        Pads a point spread funciton with zeros. The original PSF will
        appear in the upper-lerft corner in order to preserve center
        of PSF for future computations.
    '''
    zeros = np.zeros(shape)
    zeros[0:mat.shape[0], 0:mat.shape[1]] = mat
    return zeros

def compute_error(x1, x2):
    '''
        Returns the Frobenian normalization of the form ||X1 - X2||_2.

        Parameters:
            - x1 (mat): First matrix.
            - x2 (mat): Second matrix to subtract from X1.
    '''
    return np.linalg.norm(x1-x2, ord='fro')

def least_squares(x1, x2, x3):
    '''
        Returns least-squares normalization of the form ||AX - B||_2 + ||X||_2

        Parameters:
            - x1 (mat): matrix of the form AX
            - x2 (mat): matrix of the form B
            - x3 (mat): matrix of hte form X

        Returns:
            - Number representing norm ||AX - B||_2 + ||X||_2
    '''
    return np.linalg.norm(x1-x2, ord='fro')**2 + np.linalg.norm(x3)**2

def make_image(m: int, n: int):
    ''''
        Creates an image with squares and circles of dimensions (m, n).

        Parameters:
            - m (int): size of image in pixels in x axis
            - n (int): size of image in pixels in y axis

        Returns:
            - matrix with the given shapes.
    '''
    X = np.zeros((m, n))
    # add 1 since range will stop just before
    I = np.arange(round(m/5), round(3*m/5) + 1)
    J = np.arange(round(m/5), round(3*m/5) + 1)

    # add rectangle to image
    X[I.min():I.max(), J.min():J.max()] = 0.5
    for i in range(0, m): # arrays are zero indexed in matlab
        for j in range(0, n):
            # add circle to image
            x_val = (i - round(3*m/5)) ** 2
            y_val = (j - round(5*n/8)) ** 2
            if x_val + y_val < round(max(m, n)/5) ** 2:
                X[i, j] = 1
    return X