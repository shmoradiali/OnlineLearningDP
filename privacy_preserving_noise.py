import numpy as np


def get_noise_vector(dim, D):
        return np.array([D() for _ in range(dim)])


def get_laplace_dist(b):
    def sample_noise():
        return np.random.laplace(0, b)
    return sample_noise


def get_gaussian_dist(b):
    def sample_noise():
        return np.random.normal(0, b)
    return sample_noise
