from math import ceil
import numpy as np
from binary_tree_mechanism import BinaryTreeMechanism
from privacy_preserving_noise import get_noise_vector


class AgarwalSinghPrivateFTRL:
    def __init__(self, dim, T, D, oracle, learning_rate) -> None:
        self.dim = dim
        self.T = T
        self.D = D
        self.oracle = oracle
        self.rate = learning_rate
        self.tree = BinaryTreeMechanism(dim, T, D)
        self.L = np.zeros(dim)
        for _ in range(ceil(np.log2(T))):
            self.L += get_noise_vector(dim, D)
    
    def predict(self, t):
        return self.oracle(self.L, self.rate)
    
    def observe_loss(self, t, l_t):
        self.tree.update(t, l_t)
        self.L = self.tree.private_partial_sum(t)
