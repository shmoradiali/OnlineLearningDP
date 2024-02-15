import numpy as np
from privacy_preserving_noise import get_noise_vector


class Node:
    def __init__(self, left_child, right_child, interval, value, noise) -> None:
        self.left_child = left_child
        self.right_child = right_child
        self.interval = interval
        self.value = value
        self.noise = noise


class BinaryTreeMechanism:
    def _construct_tree(self, root) -> Node:
        lf, rg = root.interval
        if rg - lf == 1:
            return root
        mid = (lf + rg) // 2
        left_root = Node(None, None, (lf, mid), np.zeros(self.dim), get_noise_vector(self.dim, self.D))
        right_root = Node(None, None, (mid, rg), np.zeros(self.dim), get_noise_vector(self.dim, self.D))
        return Node(self._construct_tree(left_root),
                    self._construct_tree(right_root),
                    root.interval, 
                    np.zeros(self.dim),
                    root.noise)

    def __init__(self, dim, T, D) -> None:
        self.dim = dim
        self.T = T
        self.D = D
        self.root = self._construct_tree(Node(None, None, (1, T + 1), np.zeros(self.dim), get_noise_vector(self.dim, self.D)))
        self.array = np.zeros((T + 1, dim))
        self.last_update = 0
    
    def _update(self, t, val, cur) -> None:
        if cur is None or t < cur.interval[0] or t >= cur.interval[1]:
            return
#        print(f"Updating node [{cur.interval[0]}, {cur.interval[1]})")
        cur.value += val - self.array[t]
        self._update(t, val, cur.left_child)
        self._update(t, val, cur.right_child)

    def update(self, t, val) -> None:
        if t <= self.last_update:
            raise Exception(f"Attempted updating {t} when the last updated index is {self.last_update}. Updating previous values violates privacy!")
        self.last_update = t
        cur = self.root
        self._update(t, val, cur)
        self.array[t] = val

    def _get_partial_sum(self, t, cur):
        if cur.interval[1] <= t:
#            print("--------")
#            print(f"On node [{cur.interval[0]}, {cur.interval[1]}) sum is: ")
#            print(cur.value)
#            print(cur.noise)
#            print("--------")
            return cur.value + cur.noise
        if cur is None or cur.interval[0] >= t:
            return np.zeros(self.dim)

        res = np.zeros(self.dim)
        res += self._get_partial_sum(t, cur.left_child)
        res += self._get_partial_sum(t, cur.right_child)
        return res
    
    def private_partial_sum(self, t):
        cur = self.root
        return self._get_partial_sum(t, cur)
