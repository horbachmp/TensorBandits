import numpy as np
from itertools import product
from tqdm import tqdm
import random

from utils.tensor import *
from utils.bandit import *

class TensorEpochGreedy:
    def __init__(self, dimensions, ranks, num_context_dims, bandit, total_steps=5000, explore_steps=1000, lambda1=20.0, lambda2=20.0, conf_int_len=0.1, img_name=None) -> None:
        self.bandit = bandit
        self.dimensions = dimensions
        self.num_context_dims = num_context_dims
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
        self.img_name = img_name

    def ExploreStep(self, arm):
        self.steps_done += 1
        reward = self.bandit.PlayArm(arm, arm[:self.num_context_dims])
        arm_tensor = np.zeros(self.dimensions, dtype=int)
        arm_tensor[tuple(arm)] = 1
        self.Reward_vec_sum += arm_tensor * reward
        self.have_info = self.have_info | arm_tensor
        self.num_pulls += arm_tensor


    def ExploitStep(self, arm):
        self.steps_done += 1
        reward = self.bandit.PlayArm(arm, arm[:self.num_context_dims])
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


    def FindBestCurrArm(self, context_ind):
        ind = np.argmax(self.Reward_vec_est[context_ind])
        ind = np.unravel_index(ind, self.Reward_vec_est[context_ind].shape)
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
            self.ExploreStep(arm)
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
            context_dims = self.dimensions[:self.num_context_dims]
            context = np.array([random.randint(0, dim - 1) for dim in context_dims])
            current_arm = self.FindBestCurrArm(context)
            current_arm_tensor = self.CreateArmTensorByIndex(current_arm)
            reward = self.ExploitStep(current_arm)
            self.UpdateEstimation()
        # self.bandit.PlotRegret(self.img_name, algo_name="Epoch Greedy", plot_random=False)



def main():
    # seed = 42
    # np.random.seed(seed)
    X = np.array([    # title, subtitle, picture
            [[2.9, 2.3, 2.3],
             [2.7, 2.1, 2.1],
             [3.6, 2.4, 2.4]],
            [[3.4, 3.4, 2.8],
             [3.2, 2.6, 2.6],
             [3.5, 2.9, 2.9]],
            [[1.4, 0.8, 0.8],
             [1.2, 0.6, 0.6],
             [1.5, 0.9, 0.9 ]]])
    bandit = TensorBandit(X, 0.5)
    algo = TensorEpochGreedy(dimensions=[3,3,3], ranks=[2,2,2], num_context_dims=1, bandit=bandit, img_name='/home/maryna/HSE/Bandits/TensorBandits/low_rank_algs/pictures/algs_new/epoch_greedy_only.png')
    algo.PlayAlgo()
    
    

if __name__ == "__main__":
    main()