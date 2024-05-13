import numpy as np
from tensorly.decomposition import tensor_train
from tqdm import tqdm

from utils.bandit import TensorBandit
from utils.tt_utils import optima_tt_max, tt_sum, get_tensor_from_tt
from utils.tensor import silrtc, Tensor


class TensorTrainAlgo:
    def __init__(self, dimensions, ranks, bandit, total_steps=500, explore_steps=200, k=10, update_each=50, img_name=None) -> None:
        self.bandit  = bandit
        self.dimensions = dimensions
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
        reward = self.bandit.PlayArm(arm)
        arm_tensor = np.zeros(self.dimensions, dtype=int)
        arm_tensor[tuple(arm)] = 1
        self.num_pulls += arm_tensor
        self.Reward_vec_sum += arm_tensor * reward
        return reward

    def CreateArmTensorByIndex(self, ind):
        arm_tensor = np.zeros(self.dimensions, dtype=int)
        arm_tensor[ind] = 1
        return arm_tensor


    def FindBestCurrArm(self):
        arm = tuple(optima_tt_max(self.cores, self.k, self.ranks))
        return arm


    def UpdateEstimation(self):
        current_estimation = self.Reward_vec_sum / np.where(self.num_pulls == 0, 1, self.num_pulls)
        # current_estimation = get_tensor_from_tt(self.cores)
        current_estimation = silrtc(Tensor(current_estimation), omega=np.where(self.num_pulls > 0, 1, 0)).data
        self.cores = tensor_train(current_estimation, rank=self.ranks)



    def PlayAlgo(self):
        for step in range(self.explore_steps):
            self.curr_step += 1
            arm = np.random.randint(0, high=self.dimensions, size=len(self.dimensions))
            reward = self.Step(arm)
        estimation = self.Reward_vec_sum / np.where(self.num_pulls == 0, 1, self.num_pulls)
        estimation = silrtc(Tensor(estimation), omega=np.where(self.num_pulls > 0, 1, 0)).data
        self.cores = tensor_train(estimation, rank=self.ranks)

        for step in tqdm(range(self.explore_steps + 1, self.total_steps + 1)):
            self.curr_step += 1
            current_arm = self.FindBestCurrArm()
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
            
            
        best_arm = self.FindBestCurrArm()
        print("Best arm:", best_arm)
        print(self.bandit.X[best_arm])

        # self.bandit.PlotRegret(self.img_name)






def main():
    seed = 42
    np.random.seed(seed)
    # X = np.array([    # title, subtitle, picture
    #         [[2.9, 2.3, 2.3],
    #          [2.7, 2.1, 2.1],
    #          [3.6, 2.4, 2.4]],
    #         [[3.4, 3.4, 2.8],
    #          [3.2, 2.6, 2.6],
    #          [3.5, 2.9, 2.9]],
    #         [[1.4, 0.8, 0.8],
    #          [1.2, 0.6, 0.6],
    #          [1.5, 0.9, 0.9 ]]])
    dimensions = [10, 10, 10]
    X = np.random.rand(*dimensions)
    bandit = TensorBandit(X, 0.5)
    algo = TensorTrainAlgo(dimensions=dimensions, ranks=[1,3,3,1], bandit=bandit)
    algo.PlayAlgo()
    real_best = np.unravel_index(np.argmax(X), X.shape)
    print("real best arm:",  real_best)
    print(X[real_best])
    
    

if __name__ == "__main__":
    main()