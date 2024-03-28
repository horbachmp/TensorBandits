import numpy as np
from tensorly.decomposition import tucker
import tensorly as tl
import math
from scipy.linalg import null_space
from itertools import product
from sklearn.linear_model import Ridge
import itertools


from utils.tensor import *
from utils.bandit import *



class TensorElimination:
    def __init__(self, dimensions, ranks, bandit, total_steps=20000, explore_steps=200, lambda1=0.01, lambda2=20.0, conf_int_len=0.3, img_name=None, update_arm_on_step=None, delete_arm_on_step=None) -> None:
        self.bandit  = bandit
        self.dimensions = dimensions
        self.ranks = ranks
        self.total_steps = total_steps
        self.explore_steps = explore_steps
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.conf_int_len = conf_int_len
        self.Reward_vec_est = np.zeros(self.dimensions)
        self.Reward_vec_sum = np.zeros(self.dimensions)
        self.Reward_vec_est_UUT = np.zeros(self.dimensions)
        self.have_info = np.zeros(self.dimensions).astype(bool)
        self.num_pulls = np.zeros(self.dimensions)
        self.all_arms = set(product(*list(map(lambda x: list(range(x)), self.dimensions))))
        self.factors = list()
        self.perp_factors = list()
        self.q = 0
        arr = np.concatenate((np.full(self.q, lambda1), np.full((np.prod(self.dimensions)) - self.q, lambda2)))
        self.V_t = np.diag(arr)
        self.V_t_inv = np.linalg.inv(self.V_t)
        self.steps_done = 0
        self.curr_beta = 0
        self.img_name = img_name
        self.update_arm_on_step = update_arm_on_step
        self.delete_arm_on_step = delete_arm_on_step

    def Step(self, arm):
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
            vec_e_U = (np.concatenate((self.factors[i], self.perp_factors[i]), axis=1).T) @ vec_e
            arm = np.outer(arm, vec_e_U)
        if vectorise:
            return arm.ravel().reshape(-1,1)
        return arm


    def FindBestCurrArm(self):
        norms = list()
        arms = list(self.all_arms)
        for arm in arms:
            arm_vec = self.CreateArmTensorByIndex(arm)
            norms.append(np.sqrt((arm_vec.T) @ self.V_t_inv @ (arm_vec)))
        return arms[np.argmax(norms)]


    def UpdateEstimation(self):
        Rew_vec_ini = self.Reward_vec_sum / self.num_pulls
        Rew_vec_completed = silrtc(Tensor(Rew_vec_ini), omega=self.have_info)
        Rew_vec_completed = Rew_vec_completed.data
        core, factors = tucker(Rew_vec_completed, rank=self.ranks)
        self.factors = factors
        perp_factors = []
        ind = 0
        self.Reward_vec_est = Rew_vec_completed
        self.Reward_vec_est_UUT = Rew_vec_completed
        for factor in factors:
            perp_factor = null_space(factor.T)
            perp_factors.append(perp_factor)
            vec_e_U = (np.concatenate((factor, perp_factor), axis=1).T)
            self.Reward_vec_est_UUT = marginal_multiplication(self.Reward_vec_est_UUT, vec_e_U, ind)
            ind += 1
        self.q = np.prod(self.dimensions) - np.prod(np.array(self.dimensions) - np.array(self.ranks))
        arr = np.concatenate((np.full(self.q, self.lambda1), np.full((np.prod(self.dimensions)) - self.q, self.lambda2)))
        self.V_t = np.diag(arr)
        self.V_t_inv = np.linalg.inv(self.V_t)
        self.perp_factors = perp_factors


    def FindBestBeta(self, arm_tesnors, rewards):
        new_arm_tesnors = np.array(arm_tesnors)
        new_arm_tesnors[:,:self.q] *= self.lambda1
        new_arm_tesnors[:,self.q:] *= self.lambda2
        ridge_reg = Ridge(alpha=1, fit_intercept=False)
        ridge_reg.fit(arm_tesnors, rewards)
        self.curr_beta = ridge_reg.coef_[0]
        return self.curr_beta

    def UpdateArms(self, dim, ind, delta):
        combinations = set(product(*list(map(lambda x: list(range(x)), self.dimensions))))
        filtered_combinations = [combo for combo in combinations if combo[dim] == ind]
        for arm in filtered_combinations:
            self.bandit.UpdateArm(arm, delta)
            self.all_arms.add(arm)

    def DeleteArms(self, dim, ind):
        combinations = set(product(*list(map(lambda x: list(range(x)), self.dimensions))))
        filtered_combinations = [combo for combo in combinations if combo[dim] == ind]
        for arm in filtered_combinations:
            if arm in self.all_arms:
                self.all_arms.remove(arm)
        if len(self.all_arms) == 0:
            self.all_arms = set(product(*list(map(lambda x: list(range(x)), self.dimensions))))
            for arm in filtered_combinations:
                self.all_arms.remove(arm)



    def PlayAlgo(self):
        for step in range(self.explore_steps):
            arm = np.random.randint(0, high=self.dimensions, size=len(self.dimensions))
            while tuple(arm) not in self.all_arms:
                arm = np.random.randint(0, high=self.dimensions, size=len(self.dimensions))
            self.Step(arm)
        self.UpdateEstimation()
        updated = False
        deleted = False
        for step in range(self.explore_steps, self.total_steps, 50):
            if self.update_arm_on_step is not None and not updated and step >= self.update_arm_on_step:
                updated = True
                self.UpdateArms(2, 2, 0.6)
            if self.delete_arm_on_step is not None and not deleted and step >= self.delete_arm_on_step:
                deleted = True
                self.DeleteArms(0, 2)
            rewards = list()
            played_arms = list()
            for iter in range(50):
                current_arm = self.FindBestCurrArm()
                current_arm_tensor = self.CreateArmTensorByIndex(current_arm)
                played_arms.append(current_arm_tensor[:, 0])
                reward = self.Step(current_arm)
                rewards.append(reward)
                self.V_t += current_arm_tensor @ current_arm_tensor.T
                self.V_t_inv = np.linalg.inv(self.V_t)
            # eliminating
            self.FindBestBeta(played_arms, rewards)
            lower_bounds = list()
            upper_bounds = list()
            for arm in self.all_arms:
                arm_tensor = self.CreateArmTensorByIndex(arm)
                value = np.dot(self.curr_beta, arm_tensor)
                delta = self.conf_int_len * np.sqrt((arm_tensor.T) @ self.V_t_inv @ (arm_tensor))
                lower_bounds.append(value - delta)
                upper_bounds.append((value + delta, arm))
            curr_max = np.max(lower_bounds)
            # print("БЫЛО РУЧЕК", len(self.all_arms))
            # print(self.all_arms)
            # print(curr_max)
            self.all_arms = {pair[1] for pair in upper_bounds if pair[0] > curr_max}
            # print("СТАЛО РУЧЕК", len(self.all_arms))
            # if (len(self.all_arms) == 1):
            #     print("-------------------------------------------------------------------------------------------------------------------------------")
            #     print("Лучшая ручка:", self.all_arms)
            #     print("Затрачено шагов:", self.steps_done)
                # break
            
            self.UpdateEstimation()
        best_arm = self.FindBestCurrArm()
        print("Best combination: title -", best_arm[0]+1, "subtitle -", best_arm[1] + 1, "picture -", best_arm[2] + 1)
        


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
    algo = TensorElimination(dimensions=[3,3,3], ranks=[2,2,2], bandit=bandit, img_name="elimination_upd4000_del6000", delete_arm_on_step=6000, update_arm_on_step=4000)
    algo.PlayAlgo()
    
    

    

main()