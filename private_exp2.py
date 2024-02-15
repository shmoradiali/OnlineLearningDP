from random import choices
import numpy as np
from privacy_preserving_noise import get_noise_vector


class AgarwalSinghPrivateEXP2:
    def __init__(self, dim, T, D, learning_rate, gamma, mu) -> None:
        self.dim = dim
        self.T = T
        self.D = D
        self.rate = learning_rate
        self.gamma = gamma
        self.mu = mu
        self.w_t = np.ones(dim)
    
    def predict(self, t):
        q_t = self.w_t / np.sum(self.w_t)
        p_t = (1 - self.gamma) * q_t + self.gamma * self.mu
        self.i_t = choices(range(self.dim), p_t)[0]
        self.p_it = p_t[self.i_t]
        return self.i_t
    
    def observe_loss(self, t, val):
        zval = val + self.D()
        self.w_t[self.i_t] *= np.exp(-self.rate * zval / self.p_it)
