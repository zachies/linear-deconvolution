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

def test_add_noise_zero():
    # arrange
    mat = np.zeros((32, 32))

    # act
    noise = add_noise(mat, 0)

    # assert
    # - check for new instance
    assert mat is not noise
    # - check that matrix is the same
    assert np.linalg.norm(mat) == np.linalg.norm(noise)

def test_add_noise_valid():
    # arrange
    mat = np.zeros((32, 32))

    # act
    noise = add_noise(mat, 1)

    # assert
    # - check for new instance
    assert mat is not noise
    # - check that matrix is different
    assert np.linalg.norm(mat) != np.linalg.norm(noise)