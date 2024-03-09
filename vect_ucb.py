import numpy as np
from tensorly.decomposition import tucker
import tensorly as tl
import math
from scipy.linalg import null_space
from itertools import product
from sklearn.linear_model import Ridge

from utils.tensor import *
from utils.bandit import *


class Vect_UCB_1():
    def __init__(self, bandit, total_steps=3000, explore_steps=200) -> None:
        self.bandit  = bandit
        self.total_steps = total_steps
        self.explore_steps = explore_steps
        self.steps_done = 0
        self.curr_beta = 0
        self.Reward_vec_sum = np.zeros(self.bandit.dimensions)
        self.have_info = np.zeros(self.bandit.dimensions).astype(bool)
        self.num_pulls = np.zeros(self.bandit.dimensions)

    def ExploreStep(self, arm):
        self.steps_done += 1
        reward = self.bandit.PlayArm(arm)
        print(reward)
        arm_tensor = np.zeros(self.bandit.dimensions, dtype=int)
        arm_tensor[tuple(arm)] = 1
        self.Reward_vec_sum += arm_tensor * reward
        self.have_info = self.have_info | arm_tensor
        self.num_pulls += arm_tensor


    def ExploitStep(self, arm):
        self.steps_done += 1
        reward = self.bandit.PlayArm(arm)
        print(reward)
        arm_tensor = np.zeros(self.bandit.dimensions, dtype=int)
        arm_tensor[tuple(arm)] = 1
        self.Reward_vec_sum += arm_tensor * reward
        self.have_info = self.have_info | arm_tensor
        self.num_pulls += arm_tensor
        return reward
    

    def FindBestCurrArm(self):
        estimation = self.Reward_vec_sum / self.num_pulls + np.sqrt(2 * np.log(self.steps_done) / self.num_pulls)
        index = np.unravel_index(np.argmax(estimation), estimation.shape)
        return index
    
    def PlayAlgo(self):
        for step in range(self.explore_steps):
            arm = np.random.randint(0, high=self.bandit.dimensions, size=len(self.bandit.dimensions))
            print(arm)
            self.ExploreStep(arm)
        for step in range(self.total_steps - self.explore_steps):
            arm = self.FindBestCurrArm()
            self.ExploitStep(arm)
        print(self.FindBestCurrArm())
        estimation = self.Reward_vec_sum / self.num_pulls + np.sqrt(2 * np.log(self.steps_done) / self.num_pulls)
        print("real argmax", np.unravel_index(np.argmax(estimation), estimation.shape))
        print("real R", self.bandit.X)
        print("estimated", estimation)





def main():
    # seed = 42
    # np.random.seed(seed)
    bandit = TensorBandit(dimensions=[5,5], ranks=[3,3])
    algo = Vect_UCB_1(bandit=bandit)
    algo.PlayAlgo()
    
    # bandit = Bandit([3,3], [2,2])
    

main()