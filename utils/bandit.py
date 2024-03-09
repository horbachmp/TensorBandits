from utils.tensor import *
from tensorly.decomposition import tucker
import numpy as np

class TensorBandit:
    def __init__(self, dimensions, ranks) -> None:
        self.dimensions = np.array(dimensions)
        self.ranks = ranks
        self.X = np.random.rand(*self.dimensions)
        core, factors = tucker(self.X, rank=self.ranks)
        X = core
        ind = 0
        for factor in factors:
            X = marginal_multiplication(X, factor, ind)
            ind += 1
        self.X = X



    def PlayArm(self, index):
        arm_tensor = np.zeros(self.dimensions, dtype=int)
        arm_tensor[tuple(index)] = 1
        determ_reward = np.sum(self.X * arm_tensor)
        noise = np.random.normal(0, 1, 1)
        print(determ_reward, noise[0])
        return np.array([determ_reward + noise[0]])