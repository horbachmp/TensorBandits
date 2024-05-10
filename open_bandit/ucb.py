import numpy as np
from itertools import product

from utils.tensor import *
from utils.bandit import *


class Vect_UCB_1():
    def __init__(self, dimensions, bandit, total_steps=20000, explore_steps=200, img_name=None, update_arm_on_step=None, delete_arm_on_step=None) -> None:
        self.dimensions = dimensions
        self.bandit  = bandit
        self.total_steps = total_steps
        self.explore_steps = explore_steps
        self.steps_done = 0
        self.curr_beta = 0
        self.Reward_vec_sum = np.zeros(self.dimensions)
        self.have_info = np.zeros(self.dimensions).astype(bool)
        self.num_pulls = np.zeros(self.dimensions)
        self.img_name = img_name
        self.update_arm_on_step = update_arm_on_step
        self.delete_arm_on_step = delete_arm_on_step
        self.all_arms = set(product(*list(map(lambda x: list(range(x)), self.dimensions))))

    def Step(self, arm):
        self.steps_done += 1
        reward = self.bandit.PlayArm(arm)
        arm_tensor = np.zeros(self.dimensions, dtype=int)
        arm_tensor[tuple(arm)] = 1
        self.Reward_vec_sum += arm_tensor * reward
        self.have_info = self.have_info | arm_tensor
        self.num_pulls += arm_tensor
        return reward
    
    def UpdateArms(self, dim, ind, delta):
        combinations = set(product(*list(map(lambda x: list(range(x)), self.dimensions))))
        filtered_combinations = [combo for combo in combinations if combo[dim] == ind]
        for arm in filtered_combinations:
            self.bandit.UpdateArm(arm, delta)
            self.num_pulls[tuple(arm)] = 0

    def DeleteArms(self, dim, ind):
        combinations = set(product(*list(map(lambda x: list(range(x)), self.dimensions))))
        filtered_combinations = [combo for combo in combinations if combo[dim] == ind]
        for arm in filtered_combinations:
            self.Reward_vec_sum[tuple(arm)] = -1e9
            self.bandit.X[tuple(arm)] = -1e9
            if arm in self.all_arms:
                self.all_arms.remove(arm)

    def FindBestCurrArm(self):
        estimation = self.Reward_vec_sum / (self.num_pulls + 1e-9) + np.sqrt(2 * np.log(self.steps_done) / (self.num_pulls + 1e-9))
        index = np.unravel_index(np.argmax(estimation), estimation.shape)
        return index
    

    def GetArmsRatings(self, unknown_threshold=7):
        estimation = self.Reward_vec_sum / (self.num_pulls + 1e-9)
        sorted_indices = [tuple(np.unravel_index(x, estimation.shape)) for x in np.argsort(estimation, axis=None)]
        is_determined = self.num_pulls > unknown_threshold
        unknown = np.where(is_determined == False)
        unknown_indices = set(zip(*unknown))
        ratings = {}
        filtered_indices = []
        for arm in sorted_indices:
            if arm in unknown_indices:
                ratings[arm] = 0
            else:
                filtered_indices.append(arm)
        length = len(filtered_indices)
        one_class_num = length // 5
        curr_class = 5
        curr_class_num = one_class_num
        for i in range(len(filtered_indices) - 1, -1, -1):
            ratings[filtered_indices[i]] = curr_class
            curr_class_num -= 1
            if curr_class_num == 0 and curr_class > 1:
                curr_class -= 1
                curr_class_num = one_class_num
        print(ratings)

    
    def PlayAlgo(self):
        for step in range(self.explore_steps):
            arm = np.random.randint(0, high=self.dimensions, size=len(self.dimensions))
            while tuple(arm) not in self.all_arms:
                arm = np.random.randint(0, high=self.dimensions, size=len(self.dimensions))
            self.Step(arm)
        updated = False
        deleted = False
        for step in range(self.explore_steps, self.total_steps):
            if self.update_arm_on_step is not None and not updated and step >= self.update_arm_on_step:
                updated = True
                self.UpdateArms(2, 2, 0.6)
            if (self.delete_arm_on_step is not None) and (not deleted) and step >= self.delete_arm_on_step:
                deleted = True
                self.DeleteArms(0, 2)
            arm = self.FindBestCurrArm()
            self.Step(arm)
        best_arm = self.FindBestCurrArm()
        print("Best combination: title -", best_arm[0]+1, "subtitle -", best_arm[1] + 1, "picture -", best_arm[2] + 1)
        self.GetArmsRatings()
        self.bandit.PlotRegret(self.img_name)

def main():
    seed = 42
    np.random.seed(seed)
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
    algo = Vect_UCB_1(dimensions=[3,3,3], bandit=bandit, img_name="ucb_upd4000", update_arm_on_step=4000)
    algo.PlayAlgo()

if __name__ == "__main__":
    main()