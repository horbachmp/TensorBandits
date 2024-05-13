import numpy as np
from tensorly.decomposition import tensor_train
from tqdm import tqdm
import random
import itertools

from utils.bandit import OpenBanditSimulator
from utils.tt_utils import optima_tt_max, tt_sum, tt_elementwise_product, get_tensor_from_tt
from utils.tensor import silrtc, Tensor



class TensorTrainAlgo:
    def __init__(self, dimensions, ranks, num_context_dims, bandit, total_steps=500, explore_steps=200, k=10, update_each=50, img_name=None) -> None:
        self.bandit  = bandit
        self.dimensions = dimensions
        self.num_context_dims = num_context_dims
        self.ranks = ranks
        self.total_steps = total_steps
        self.explore_steps = explore_steps
        self.k = k
        self.cores = []
        self.num_pulls = np.zeros(self.dimensions)
        self.Reward_vec_sum = np.zeros(self.dimensions)
        self.curr_step = 0
        self. update_each =  update_each
        self.img_name = img_name

    def Step(self, arm):
        reward = self.bandit.PlayArm(tuple(arm))
        arm_tensor = np.zeros(self.dimensions, dtype=int)
        arm_tensor[tuple(arm)] = 1
        self.num_pulls += arm_tensor
        self.Reward_vec_sum += arm_tensor * reward
        return reward

    def CreateArmTensorByIndex(self, ind):
        arm_tensor = np.zeros(self.dimensions, dtype=int)
        arm_tensor[ind] = 1
        return arm_tensor


    def FindBestCurrArm(self, context):
        context_tensor = np.zeros(self.dimensions, dtype=int)
        context_tensor[context] = 1
        context_cores = tensor_train(context_tensor, rank=1)
        new_cores = tt_elementwise_product(self.cores, context_cores)
        arm = tuple(optima_tt_max(new_cores, self.k, self.ranks))
        return arm


    def UpdateEstimation(self):
        current_estimation = self.Reward_vec_sum / np.where(self.num_pulls == 0, 1, self.num_pulls)
        # current_estimation = get_tensor_from_tt(self.cores)
        current_estimation = silrtc(Tensor(current_estimation), omega=np.where(self.num_pulls > 0, 1, 0)).data
        self.cores = tensor_train(current_estimation, rank=self.ranks)



    def PlayAlgo(self):
        all_combinations = list(itertools.product(*[range(dim) for dim in self.dimensions[self.num_context_dims:]]))
        for step in range(self.explore_steps):
            self.curr_step += 1
            np.random.shuffle(all_combinations)
            arm = all_combinations[0]
            context = self.bandit.GetContext(self.dimensions[:self.num_context_dims])
            arm = np.concatenate([context, arm])
            reward = self.Step(arm)
        estimation = self.Reward_vec_sum / np.where(self.num_pulls == 0, 1, self.num_pulls)
        estimation = silrtc(Tensor(estimation), omega=np.where(self.num_pulls > 0, 1, 0)).data
        self.cores = tensor_train(estimation, rank=self.ranks)

        for step in tqdm(range(self.explore_steps + 1, self.total_steps + 1)):
            self.curr_step += 1
            context = self.bandit.GetContext(self.dimensions[:self.num_context_dims])
            current_arm = self.FindBestCurrArm(context)
            current_arm_tensor = self.CreateArmTensorByIndex(current_arm)
            old_val = 0
            if self.num_pulls[current_arm] != 0:
                old_val = self.Reward_vec_sum[current_arm]/ self.num_pulls[current_arm]
            reward = self.Step(current_arm)
            new_val = self.Reward_vec_sum[current_arm]/ self.num_pulls[current_arm]
            delta = new_val - old_val
            delta_cores = tensor_train(current_arm_tensor * delta, rank=1)
            self.cores = tt_sum(self.cores, delta_cores)
            if step % self.update_each == 0:
                self.UpdateEstimation()
            





def main():
    seed = 42
    np.random.seed(seed)
    num_context_dims=4
    num_arms = 5
    total_steps=5000
    explore_steps = 1000
    bandit = OpenBanditSimulator(num_context_dims, num_arms, 2 * total_steps)
    dimensions=[4,4,4,4,num_arms]
    ranks=[1,2,2,2,2,1]
    algo = TensorTrainAlgo(dimensions=dimensions, ranks=[1,2,2,2,2,1], num_context_dims=1, bandit=bandit, total_steps=total_steps, explore_steps=explore_steps) # ranks should be of len(dims) + 1 and starts and ends with 1
    algo.PlayAlgo()
    
    

if __name__ == "__main__":
    main()