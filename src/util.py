import numpy as np

from PIL import Image

def load_img(path: str) -> np.ndarray:
    '''
    Loads an image from the given path as a matrix.

    Parameters:
        path (str): path to file to open

    Returns:
        Matrix with values normalized between [0, 1]
    '''
    mat = np.asarray(Image.open(path))
    return mat / 255

def normalize(data: np.ndarray) -> np.ndarray:
    '''
    Normalizes a matrix's values into the range [0, 1].
    '''
    return (data-np.min(data))/(np.max(data)-np.min(data))