import numpy as np
from itertools import product
from tqdm import tqdm
import random
from tensorly.decomposition import tucker
import itertools

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
        reward = self.bandit.PlayArm(tuple(arm))
        # print("reward", reward)
        arm_tensor = np.zeros(self.dimensions, dtype=int)
        arm_tensor[tuple(arm)] = 1
        self.Reward_vec_sum += arm_tensor * reward
        self.have_info = self.have_info | arm_tensor
        self.num_pulls += arm_tensor


    def ExploitStep(self, arm):
        self.steps_done += 1
        reward = self.bandit.PlayArm(tuple(arm))
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
        all_combinations = list(itertools.product(*[range(dim) for dim in self.dimensions[self.num_context_dims:]]))
        #exploration
        for step in range(self.explore_steps):
            np.random.shuffle(all_combinations)
            arm = all_combinations[0]
            context = self.bandit.GetContext(self.dimensions[:self.num_context_dims])
            arm = np.concatenate([context, arm])
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
            context = self.bandit.GetContext(self.dimensions[:self.num_context_dims])
            current_arm = self.FindBestCurrArm(context)
            current_arm_tensor = self.CreateArmTensorByIndex(current_arm)
            reward = self.ExploitStep(current_arm)
            self.UpdateEstimation()
        
        # self.bandit.PlotRegret(self.img_name, algo_name="Epoch Greedy", plot_random=False)



def main():
    seed = 42
    np.random.seed(seed)
    num_context_dims=4
    num_arms = 5
    total_steps=5000
    explore_steps = 1000
    bandit = OpenBanditSimulator(num_context_dims, num_arms, 2 * total_steps)
    dimensions=[4,4,4,4,num_arms]
    ranks=[2,2,2,2,2]
    algo = TensorEpochGreedy(dimensions=dimensions, ranks=ranks, num_context_dims=1, bandit=bandit, total_steps=total_steps, explore_steps=explore_steps)
    algo.PlayAlgo()

    

if __name__ == "__main__":
    main()