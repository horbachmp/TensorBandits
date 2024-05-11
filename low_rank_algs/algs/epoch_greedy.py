import numpy as np
from itertools import product
from tqdm import tqdm

from utils.tensor import *
from utils.bandit import *

class TensorEpochGreedy:
    def __init__(self, dimensions, ranks, bandit, total_steps=3000, explore_steps=200, lambda1=20.0, lambda2=20.0, conf_int_len=0.1) -> None:
        self.bandit = bandit
        self.dimensions = dimensions
        self.total_steps = total_steps
        self.explore_steps = explore_steps
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.conf_int_len = conf_int_len
        self.ranks = ranks
        self.Reward_vec_est = np.zeros(self.dimensions)
        self.Reward_vec_sum = np.zeros(self.dimensions)
        self.have_info = np.zeros(self.dimensions).astype(bool)
        self.num_pulls = np.zeros(self.dimensions)
        self.all_arms = set(product(*list(map(lambda x: list(range(x)), self.dimensions))))
        self.factors = list()
        self.steps_done = 0

    def ExploreStep(self, arm):
        self.steps_done += 1
        reward = self.bandit.PlayArm(arm)
        # print("reward", reward)
        arm_tensor = np.zeros(self.dimensions, dtype=int)
        arm_tensor[tuple(arm)] = 1
        self.Reward_vec_sum += arm_tensor * reward
        self.have_info = self.have_info | arm_tensor
        self.num_pulls += arm_tensor


    def ExploitStep(self, arm):
        self.steps_done += 1
        reward = self.bandit.PlayArm(arm)
        arm_tensor = np.zeros(self.dimensions, dtype=int)
        arm_tensor[tuple(arm)] = 1
        self.Reward_vec_sum += arm_tensor * reward
        self.have_info = self.have_info | arm_tensor
        self.num_pulls += arm_tensor
        return reward

    def CreateArmTensorByIndex(self, ind, vectorise=True):
        arm = 1
        for i in range(len(ind)):
            vec_e = np.zeros(self.dimensions[i])
            vec_e[ind[i]] = 1
            arm = np.outer(arm, vec_e)
        if vectorise:
            return arm.ravel().reshape(-1,1)
        return arm


    def FindBestCurrArm(self):
        ind = np.argmax(self.Reward_vec_est)
        ind = np.unravel_index(ind, self.Reward_vec_est.shape)
        return ind


    def UpdateEstimation(self):
        tmp_num_pulls = self.num_pulls
        tmp_num_pulls[tmp_num_pulls == 0] = 1
        Rew_vec_ini = self.Reward_vec_sum / tmp_num_pulls
        Rew_vec_completed = silrtc(Tensor(Rew_vec_ini), omega=self.have_info)
        Rew_vec_completed = Rew_vec_completed.data
        core, factors = tucker(Rew_vec_completed, rank=self.ranks)
        self.factors = factors
        ind = 0
        self.Reward_vec_est = core
        for factor in factors:
            self.Reward_vec_est = marginal_multiplication(self.Reward_vec_est, factor, ind)
            ind += 1
        


    def PlayAlgo(self):
        #exploration
        for step in range(self.explore_steps):
            arm = np.random.randint(0, high=self.dimensions, size=len(self.dimensions))
            # print("curr_arm", arm)
            self.ExploreStep(arm)
            # print("Estimation", self.Reward_vec_est)
        tmp_num_pulls = self.num_pulls
        tmp_num_pulls[tmp_num_pulls == 0] = 1
        Rew_vec_ini = self.Reward_vec_sum / tmp_num_pulls
        Rew_vec_completed = silrtc(Tensor(Rew_vec_ini), omega=self.have_info)
        Rew_vec_completed = Rew_vec_completed.data
        core, factors = tucker(Rew_vec_completed, rank=self.ranks)
        self.factors = factors
        ind = 0
        self.Reward_vec_est = core
        for factor in factors:
            self.Reward_vec_est = marginal_multiplication(self.Reward_vec_est, factor, ind)
            ind += 1
        # reduction
        for k in tqdm(range(self.explore_steps, self.total_steps)):

            current_arm = self.FindBestCurrArm()
            current_arm_tensor = self.CreateArmTensorByIndex(current_arm)
            reward = self.ExploitStep(current_arm)
            self.UpdateEstimation()
        # print(self.bandit.X)
        # print(np.argmax(self.bandit.X), )
        index = np.unravel_index(np.argmax(self.bandit.X), self.bandit.X.shape)
        best_arm = self.FindBestCurrArm()
        print("best arm:", best_arm)
        print(self.bandit.X[best_arm])
        
        # self.bandit.PlotRegret()



def main():
    # seed = 42
    # np.random.seed(seed)
    bandit = TensorBandit(dimensions=[5,5], ranks=[3,3])
    algo = TensorEpochGreedy(bandit)
    algo.PlayAlgo()
    
    # bandit = Bandit([3,3], [2,2])
    

if __name__ == "__main__":
    main()