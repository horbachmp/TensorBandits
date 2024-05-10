import numpy as np
from tensorly.decomposition import tucker
import torch
import tensorly as tl
import math
from scipy.linalg import null_space
from itertools import product
from sklearn.linear_model import Ridge
import itertools
import random
from copy import deepcopy


from utils.tensor import *
from utils.bandit import *

def normalize(matrix):
        norm = np.sqrt(np.sum(matrix**2))
        if norm == 0:
            return matrix
        else:
            return matrix / norm

class EnsembleSampling:
    def __init__(self, dimensions, ranks, bandit, num_context_dims, prior_mus, prior_sigmas, perturb_noise, num_models=5, total_steps=20000, img_name=None, update_arm_on_step=None, delete_arm_on_step=None, print_every=100) -> None:
        self.bandit  = bandit
        self.dimensions = dimensions
        self.ranks = ranks
        self.num_context_dims = num_context_dims
        self.prior_mus = prior_mus
        self.prior_sigmas = prior_sigmas
        self.perturb_noise = perturb_noise
        self.num_models = num_models
        self.models = []
        self.zero_step_models = []
        self.total_steps = total_steps
        self.vs = list()
        self.arm_history = list()
        for _ in self.dimensions:
            self.arm_history.append([])
        self.reward_history = list()

        # self.conf_int_len = conf_int_len
        self.Reward_vec_est = np.zeros(self.dimensions)
        self.Reward_vec_sum = np.zeros(self.dimensions)
        # self.Reward_vec_est_UUT = np.zeros(self.dimensions)
        # self.have_info = np.zeros(self.dimensions).astype(bool)
        self.num_pulls = np.zeros(self.dimensions)
        self.print_every = print_every
        # self.img_name = img_name
        # self.update_arm_on_step = update_arm_on_step
        # self.delete_arm_on_step = delete_arm_on_step

    def Step(self, arm):
        # self.steps_done += 1
        reward = self.bandit.PlayArm(arm)
        noise = np.random.normal(0, self.perturb_noise)
        noise_reward = reward + noise
        arm_tensor = np.zeros(self.dimensions, dtype=int)
        arm_tensor[tuple(arm)] = 1
        self.Reward_vec_sum += arm_tensor *noise_reward
        self.num_pulls += arm_tensor
        return noise_reward

    def CreateArmTensorByIndex(self, ind):
        arm = np.zeros(self.dimensions)
        arm[tuple(ind)] = 1
        return arm


    def compute_v(self, curr_model, k, s):
        curr_S = curr_model[0]
        for h, curr_U in enumerate(curr_model[1:]):
            if h != k:
                row_index = self.arm_history[h][s]
                row = curr_U[row_index].reshape(1,-1)
                curr_S = marginal_multiplication(curr_S, row, h)
        return curr_S
    
    


    def PlayAlgo(self):
        # initializing models
        for m in range(self.num_models):
            model = []
            model.append(np.ones(self.ranks))
            for k in range(len(self.dimensions)):
                U_k = []
                for i in range(self.dimensions[k]):
                    row = np.random.multivariate_normal(self.prior_mus[k][i], self.prior_sigmas[k][i] * np.eye(len(self.prior_mus[k][i])))
                    U_k.append(row)
                model.append(U_k)
            self.models.append(model)
        self.zero_step_models = self.models.copy()
        #init phase
        for _ in range(10):
            arm = np.random.randint(0, high=self.dimensions, size=len(self.dimensions))
            for i, c in enumerate(arm):
                self.arm_history[i].append(c)
            perturb_reward = self.Step(arm)
            self.reward_history.append(perturb_reward[0])

        arm = np.random.randint(0, high=self.dimensions, size=len(self.dimensions))
        for i, c in enumerate(arm):
            self.arm_history[i].append(c)
        perturb_reward = self.Step(arm)
        self.reward_history.append(perturb_reward[0])
        # exploitation
        for step in range(self.total_steps):
            model_idx = random.randint(0, self.num_models - 1)
            new_models = deepcopy(self.models)
            curr_model = self.models[model_idx]     
            for U_index, curr_U in enumerate(curr_model[1:]):
                for row_index,  row in enumerate(curr_U):
                    first_part = np.eye(row.shape[0])
                    second_part = np.array(self.zero_step_models[model_idx][U_index + 1][row_index]).reshape(-1,1)
                    for s in range(len(self.reward_history)):
                        if self.arm_history[U_index][s] == row_index:
                            v = self.compute_v(curr_model, U_index, s).reshape(-1,1)
                            first_part += v @ v.T
                            second_part += v * self.reward_history[s]
                    first_part /= self.prior_sigmas[U_index][row_index] ** 2
                    second_part /= self.prior_sigmas[U_index][row_index] ** 2
                    # print("hi", first_part)
                    
                    new_models[model_idx][U_index + 1][row_index] = np.linalg.inv(first_part) @ second_part
                new_models[model_idx][U_index + 1] = np.apply_along_axis(normalize,0,new_models[model_idx][U_index + 1])
                # if U_index == 0:
                #     print(new_models[model_idx][U_index + 1])
            self.models = new_models
            
            self.Reward_vec_est = self.Reward_vec_sum / np.where(self.num_pulls == 0, 1, self.num_pulls)
            first_part = np.zeros_like(self.models[model_idx][0])
            second_part = np.zeros((first_part.shape[0], 1))
            for s in range(len(self.reward_history)):
                v = deepcopy(self.models[model_idx][1][self.arm_history[0][s]])
                for j in range(2, len(self.models[model_idx])):
                    v *= self.models[model_idx][j][self.arm_history[j - 1][s]]
                first_part += v @ v.T
                second_part += v * self.reward_history[s]
            # print(first_part)
            # print(second_part)
            new_S = np.linalg.inv(first_part) * second_part
            # new_S = np.linalg.inv(first_part)
            # print(new_S)
            # print(first_part.shape)
            # print(new_S.shape)
            # new_s = self.Reward_vec_est 
            # for U_index, curr_U in enumerate(curr_model[1:]):
            #     U = np.concatenate([row.T for row in curr_U])
            #     new_S = marginal_multiplication(new_S, U.T, U_index)
            self.models[model_idx][0] = new_S
            # generate new arm
            context_dims = self.dimensions[:self.num_context_dims]
            context = np.array([random.randint(0, dim - 1) for dim in context_dims])
            R_estim = self.models[model_idx][0]
            i = 0
            for U in self.models[model_idx][1:]:
                U_final = np.concatenate([row.T for row in U])
                R_estim = marginal_multiplication(R_estim, U_final, i)
                i += 1
            R = R_estim
            for x in context:
                R = R[x]
            arm = np.unravel_index(np.argmax(R), R.shape)
            arm = np.concatenate([context, arm])
            for i, c in enumerate(arm):
                self.arm_history[i].append(c)
            perturb_reward = self.Step(arm)
            self.reward_history.append(perturb_reward[0])
            if (step + 1) % self.print_every == 0:
                print("ITERATION", step + 1)
                print(R_estim)
                # for U in self.models[model_idx][1:]:
                #     print(np.sqrt(np.sum(U**2)))
                print(np.unravel_index(np.argmax(R_estim[0]), R_estim[0].shape))
                print(np.unravel_index(np.argmax(R_estim[1]), R_estim[1].shape))
                print(np.unravel_index(np.argmax(R_estim[2]), R_estim[2].shape))
        print(R_estim)
        print(np.unravel_index(np.argmax(R_estim[0]), R_estim[0].shape))
        print(np.unravel_index(np.argmax(R_estim[1]), R_estim[1].shape))
        print(np.unravel_index(np.argmax(R_estim[2]), R_estim[2].shape))

            


                        

    #     self.all_arms = {pair[1] for pair in upper_bounds if pair[0] > curr_max}
        #     # print("СТАЛО РУЧЕК", len(self.all_arms))
        #     # if (len(self.all_arms) == 1):
        #     #     print("-------------------------------------------------------------------------------------------------------------------------------")
        #     #     print("Лучшая ручка:", self.all_arms)
        #     #     print("Затрачено шагов:", self.steps_done)
        #         # break
            
        #     self.UpdateEstimation()
        # best_arm = self.FindBestCurrArm()
        # print("Best combination: title -", best_arm[0]+1, "subtitle -", best_arm[1] + 1, "picture -", best_arm[2] + 1)
        # self.GetArmsRatings()

        # self.bandit.PlotRegret(self.img_name)






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
    dimensions=[3,3,3]
    ranks=[2,2,2]
    # X = np.random.normal(mean, std, size=(100, 10))
    bandit = TensorBandit(X, 0.5)
    dimensions=[3,3,3]
    ranks=[2,2,2]
    mus = []
    sigmas = []
    for k in range(len(dimensions)):
        comp_mus = []
        comp_sigmas = []
        for i in range(dimensions[k]):
            row_mus = [0.] * ranks[i]
            comp_mus.append(row_mus)
            comp_sigmas.append(1.)
        mus.append(comp_mus)
        sigmas.append(comp_sigmas)
    algo = EnsembleSampling(dimensions=dimensions, ranks=ranks, bandit=bandit, num_context_dims=1, prior_mus=mus, prior_sigmas=sigmas, perturb_noise=0.1)
    algo.PlayAlgo()
    
    
if __name__ == "__main__":
    main()