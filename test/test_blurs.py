import pytest
import random

from util.blurs import *

def test_psf_gaussian_zero_dim():
    # arrange
    dim = (0, 0)
    s1 = 0
    s2 = 0
    i = 0
    j = 0

    # act

    # assert
    with pytest.raises(ValueError) as e_info:
        psf = psf_gaussian(dim, s1, s2, i, j)

def test_psf_gaussian_valid():
    # arrange
    dim = (32, 32)
    s1 = 3
    s2 = 3
    i = 15
    j = 15

    # act
    psf = psf_gaussian(dim, s1, s2, i, j)

    # assert
    assert isinstance(psf, (np.ndarray))