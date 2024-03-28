import numpy as np
from tensorly.decomposition import tucker
import tensorly as tl
import math
from scipy.linalg import null_space
from itertools import product
from sklearn.linear_model import Ridge

from utils.tensor import *
from utils.bandit import *



class TensorElimination:
    def __init__(self, bandit, total_steps=3000, explore_steps=200, lambda1=0.01, lambda2=20.0, conf_int_len=0.3) -> None:
        self.bandit  = bandit
        self.dimensions = self.bandit.dimensions
        self.total_steps = total_steps
        self.explore_steps = explore_steps
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.conf_int_len = conf_int_len
        self.ranks = self.bandit.ranks
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

    def ExploreStep(self, arm):
        self.steps_done += 1
        reward = self.bandit.PlayArm(arm)
        print(reward)
        arm_tensor = np.zeros(self.dimensions, dtype=int)
        arm_tensor[tuple(arm)] = 1
        self.Reward_vec_sum += arm_tensor * reward
        self.have_info = self.have_info | arm_tensor
        self.num_pulls += arm_tensor


    def ExploitStep(self, arm):
        self.steps_done += 1
        reward = self.bandit.PlayArm(arm)
        print(reward)
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
        print("----------------------------------------------------------------------------------", Rew_vec_completed)
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
        # print(perp_factors)
        self.q = np.prod(self.dimensions) - np.prod(np.array(self.dimensions) - np.array(self.ranks))
        arr = np.concatenate((np.full(self.q, self.lambda1), np.full((np.prod(self.dimensions)) - self.q, self.lambda2)))
        self.V_t = np.diag(arr)
        self.V_t_inv = np.linalg.inv(self.V_t)
        self.perp_factors = perp_factors


    def FindBestBeta(self, arm_tesnors, rewards):
        #временно сделаем вид, что лямбды равны и заюзаем готовую регрессию, потом напишу свою
        print(len(arm_tesnors), arm_tesnors[0].shape)
        print(len(rewards), rewards[0].shape)
        new_arm_tesnors = np.array(arm_tesnors)
        new_arm_tesnors[:,:self.q] *= self.lambda1
        new_arm_tesnors[:,self.q:] *= self.lambda2
        ridge_reg = Ridge(alpha=1, fit_intercept=False)
        ridge_reg.fit(arm_tesnors, rewards)
        self.curr_beta = ridge_reg.coef_[0]
        return self.curr_beta


    def PlayAlgo(self):
        #exploration
        for step in range(self.explore_steps):
            arm = np.random.randint(0, high=self.dimensions, size=len(self.dimensions))
            print(arm)
            self.ExploreStep(arm)
        self.UpdateEstimation()

        for k in range(0, self.total_steps, 50):
            rewards = list()
            played_arms = list()
            for iter in range(50):
                current_arm = self.FindBestCurrArm()
                current_arm_tensor = self.CreateArmTensorByIndex(current_arm)
                played_arms.append(current_arm_tensor[:, 0])
                print("CURRENT ARM", current_arm)
                reward = self.ExploitStep(current_arm)
                rewards.append(reward)
                print("reward", reward)
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
            print("БЫЛО РУЧЕК", len(self.all_arms))
            print(self.all_arms)
            print(curr_max)
            self.all_arms = {pair[1] for pair in upper_bounds if pair[0] > curr_max}
            print("СТАЛО РУЧЕК", len(self.all_arms))
            if (len(self.all_arms) == 1):
                print("-------------------------------------------------------------------------------------------------------------------------------")
                print("Лучшая ручка:", self.all_arms)
                print("Затрачено шагов:", self.steps_done)
                break
            
            self.UpdateEstimation()
        print("Finished")
        print("Real rewards tensor:", self.bandit.X)
        print("Rewards tensor estimation:", self.Reward_vec_est)
        print("All arms left", self.all_arms)
        index = np.unravel_index(np.argmax(self.bandit.X), self.bandit.X.shape)
        print("real best arm:", index, np.max(self.bandit.X))
        arm = self.FindBestCurrArm()
        print("best arm:", arm, self.bandit.X[arm])
        # if len(self.all_arms) == 1:
        print("estimated best arms:")
        for arm in self.all_arms:
            print(arm, self.bandit.X[arm])


        self.bandit.PlotRegret()









def main():
    # seed = 42
    # np.random.seed(seed)
    bandit = TensorBandit(dimensions=[5,5], ranks=[3,3])
    algo = TensorElimination(bandit)
    algo.PlayAlgo()
    
    # bandit = Bandit([3,3], [2,2])
    

main()