import numpy as np
import random
from copy import deepcopy
import obp
from obp.dataset import OpenBanditDataset, SyntheticBanditDataset, logistic_reward_function
import category_encoders as ce
import itertools
from tqdm import tqdm

from utils.tensor import *
from utils.bandit import *

def normalize(matrix):
        norm = np.sqrt(np.sum(matrix**2))
        if norm == 0:
            return matrix
        else:
            return matrix / norm
        
def find_closest_val(array, target):
    differences = np.abs(array - target)
    closest_index = np.argmin(differences)
    closest_number = array[closest_index]
    return closest_number

class EnsembleSampling:
    def __init__(self, dimensions, ranks, bandit, num_context_dims, prior_mus, prior_sigmas, perturb_noise, num_models=5, total_steps=20000, img_name=None, print_every=100) -> None:
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

        self.Reward_vec_est = np.zeros(self.dimensions)
        self.Reward_vec_sum = np.zeros(self.dimensions)
        self.num_pulls = np.zeros(self.dimensions)
        self.print_every = print_every

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
        all_combinations = list(itertools.product(*[range(dim) for dim in self.dimensions[self.num_context_dims:]]))
        for _ in range(1000):
            np.random.shuffle(all_combinations)
            arm = all_combinations[0]
            context = self.bandit.GetContext(self.dimensions[:self.num_context_dims])
            # break
            arm = np.concatenate([context, arm])
            for i, c in enumerate(arm):
                self.arm_history[i].append(c)
            perturb_reward = self.Step(arm)
            self.reward_history.append(perturb_reward)

        arm = np.random.randint(0, high=self.dimensions, size=len(self.dimensions))
        for i, c in enumerate(arm):
            self.arm_history[i].append(c)
        perturb_reward = self.Step(arm)
        self.reward_history.append(perturb_reward)
        # exploitation
        for step in tqdm(range(self.total_steps - 1000)):
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
                    
                    new_models[model_idx][U_index + 1][row_index] = (np.linalg.pinv(first_part) @ second_part)[:,0]
                new_models[model_idx][U_index + 1], _ = np.linalg.qr(new_models[model_idx][U_index + 1])
            self.models = new_models
            self.Reward_vec_est = self.Reward_vec_sum / np.where(self.num_pulls == 0, 1, self.num_pulls)
            first_part = self.Reward_vec_est
            for U_ind in range(1, len(self.models[model_idx])):
                first_part = marginal_multiplication(first_part, self.models[model_idx][U_ind].T, U_ind-1)
            new_S = first_part
            self.models[model_idx][0] = new_S
            # generate new arm
            context_dims = self.dimensions[:self.num_context_dims]
            context = np.array([random.randint(0, dim - 1) for dim in context_dims])
            R_estim = self.models[model_idx][0]
            for U_ind in range(1, len(self.models[model_idx])):
                R_estim = marginal_multiplication(R_estim, self.models[model_idx][U_ind], U_ind-1)
            R = R_estim
            for x in context:
                R = R[x]
            arm = np.unravel_index(np.argmax(R), R.shape)
            arm = np.concatenate([context, arm])
            for i, c in enumerate(arm):
                self.arm_history[i].append(c)
            perturb_reward = self.Step(arm)
            self.reward_history.append(perturb_reward)
            # if (step + 1) % self.print_every == 0:
            #     print("ITERATION", step + 1,"model num:", model_idx)
            #     print(R_estim)
            #     print(np.unravel_index(np.argmax(R_estim[0]), R_estim[0].shape))
            #     print(np.unravel_index(np.argmax(R_estim[1]), R_estim[1].shape))
            #     print(np.unravel_index(np.argmax(R_estim[2]), R_estim[2].shape))
        self.bandit.PlotRegret("/home/maryna/HSE/Bandits/TensorBandits/context_algs_open_bandit/ens_samp_vs_random.png")



class OpenBanditTest:
    def __init__(self, behavior_policy="bts", campaign="all") -> None:
        features = ['user-item_affinity_5', 'user_feature_0', 'user_feature_1', 'user_feature_2', 'timestamp', 'item_id', 'position', 'propensity_score', 'click']
        self.dataset = OpenBanditDataset(behavior_policy=behavior_policy, campaign=campaign)
        self.data = self.dataset.data[features]
        self.data = self.data.sort_values(by='timestamp', ascending=True)
        self.context = self.data[['user-item_affinity_5', 'user_feature_0', 'user_feature_1', 'user_feature_2']]
        
        encoder = ce.OrdinalEncoder(cols=['user-item_affinity_5', 'user_feature_0', 'user_feature_1', 'user_feature_2'], return_df=True)


        self.context = encoder.fit_transform(self.context)


        self.algo_info = self.data[['item_id', 'position', 'propensity_score', 'click']]
        self.data_dict = self.dataset.obtain_batch_bandit_feedback()
        # print(self.data_dict)
        self.step_index = 0
        self.behavior_reward = [0]
        self.quality = []
    
    def GetContext(self):
        return self.context.iloc[self.step_index].tolist()


    def PlayArm(self, sorted_indices):
        index = sorted_indices[:3]
        index = list(map(lambda x : x[0] + 1, index))
        print(index)
        index_dict = {index_tuple: position for position, index_tuple in enumerate(zip(*sorted_indices))}
        curr_system_arm = self.data_dict['action'][self.step_index]
        if self.data_dict['reward'][self.step_index]:
            self.quality.append((index_dict[tuple(index)]+1)/self.data_dict['pscore'][self.step_index])
        else:
            self.quality.append(0)
        while self.step_index < self.dataset.n_actions and curr_system_arm not in list(index):
            self.step_index += 1
            curr_system_arm = self.data_dict['action'][self.step_index]
            print(curr_system_arm, index)
            if self.data_dict['reward'][self.step_index]:
                self.quality.append((index_dict[tuple(index)]+1)/self.data_dict['pscore'][self.step_index])
            else:
                self.quality.append(0)
        if self.step_index >= self.dataset.n_actions:
            return None
        reward = self.data_dict['reward'][self.step_index]
        self.behavior_reward.append(self.behavior_reward[-1] + reward)
        return reward
    
class OpenBanditSimulator:
    def __init__(self, num_context_dims, num_arms, n_rounds) -> None:
        self.num_context_dims = num_context_dims
        self.dataset = SyntheticBanditDataset(
            n_actions=num_arms,
            dim_context=num_context_dims,
            reward_type="binary", # "binary" or "continuous"
            reward_function=logistic_reward_function,
            behavior_policy_function=None, # uniformly random
            random_state=12345,
        )
        bandit_feedback = self.dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
        self.context = bandit_feedback['context']
        # print(self.context.shape, np.max(self.context), np.min(self.context))
        self.exp_reward = bandit_feedback['expected_reward']
        
        self.step_index = 0
        self.reward = [0]
        self.regret = [0]
        self.random_regret = [0]
    
    def GetContext(self, dims):
        float_context = self.context[self.step_index]
        int_context = []
        for i in range(len(dims)):
            num = find_closest_val(np.array(list(range(dims[i])))- dims[i]//2, float_context[i])
            int_context.append(num + dims[i]//2)
        return int_context


    def PlayArm(self, arm):
        arm = arm[self.num_context_dims:]
        reward = self.exp_reward[self.step_index][arm][0]
        self.reward.append(self.reward[-1] + reward)
        regret = np.max(self.exp_reward[self.step_index]) - reward
        self.regret.append(self.regret[-1] + regret)
        random_arm = np.random.randint(0, len(self.exp_reward[self.step_index]))
        random_regret = np.max(self.exp_reward[self.step_index]) - self.exp_reward[self.step_index][random_arm]
        self.random_regret.append(self.random_regret[-1] + random_regret)
        self.step_index += 1
        return reward
    
    def PlotRegret(self, img_name=None):
        plt.plot(self.regret, label='Ensemble Sampling')
        plt.plot(self.random_regret, label='Random Algorithm')
        plt.xlabel('Steps')
        plt.ylabel('Regret')
        plt.title('Regret Plot for Ensemble Sampling and Random Algorithm')
        plt.legend()
        if img_name is not None:
            plt.savefig(img_name)
        else:
            plt.show()



def main():
    seed = 42
    np.random.seed(seed)
    num_context_dims=4
    num_arms = 5
    total_steps = 10000
    bandit = OpenBanditSimulator(num_context_dims, num_arms, 2 * total_steps)
    dimensions=[4,4,4,4,num_arms]
    ranks=[2,2,2,2,2]
    mus = []
    sigmas = []
    for k in range(len(dimensions)):
        comp_mus = []
        comp_sigmas = []
        for i in range(dimensions[k]):
            row_mus = [0.] * ranks[k]
            comp_mus.append(row_mus)
            comp_sigmas.append(1.)
        mus.append(comp_mus)
        sigmas.append(comp_sigmas)
    algo = EnsembleSampling(dimensions=dimensions, ranks=ranks, bandit=bandit, num_context_dims=num_context_dims, prior_mus=mus, prior_sigmas=sigmas, perturb_noise=0.1, total_steps=total_steps)
    algo.PlayAlgo()
    
    
if __name__ == "__main__":
    main()