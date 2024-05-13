from utils.tensor import *
import numpy as np
import matplotlib.pyplot as plt

import obp
from obp.dataset import OpenBanditDataset, SyntheticBanditDataset, logistic_reward_function
import category_encoders as ce

def find_closest_val(array, target):
    differences = np.abs(array - target)
    closest_index = np.argmin(differences)
    closest_number = array[closest_index]
    return closest_number

class TensorBandit:
    def __init__(self, reward_tesor, noise_level, log=False) -> None:
        self.dimensions = reward_tesor.shape
        self.X = reward_tesor
        self.noise_level = noise_level
        self.log=log
        
        self.opt_arm_rew = np.max(self.X)
        self.regrets = [0]

    def UpdateArm(self, index, delta):
        arm_tensor = np.zeros(self.dimensions, dtype=int)
        arm_tensor[tuple(index)] = delta
        self.X += arm_tensor

    def PlayArm(self, index):
        arm_tensor = np.zeros(self.dimensions, dtype=int)
        arm_tensor[tuple(index)] = 1
        determ_reward = np.sum(self.X * arm_tensor)
        noise = np.random.normal(0, self.noise_level, 1)
        reward = determ_reward + noise[0]
        if self.log:
            print(index)
            print(determ_reward, noise[0])
            print(reward)
        regret = self.opt_arm_rew - reward
        self.regrets.append(self.regrets[-1] + regret)
        return np.array([reward])
    
    def PlotRegret(self, img_name=None):
        plt.plot(self.regrets)
        if img_name is not None:
            plt.savefig(img_name)
        # plt.show()


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
        index_dict = {index_tuple: position for position, index_tuple in enumerate(zip(*sorted_indices))}
        curr_system_arm = self.data_dict['action'][self.step_index]
        if self.data_dict['reward'][self.step_index]:
            self.quality.append((index_dict[tuple(index)]+1)/self.data_dict['pscore'][self.step_index])
        else:
            self.quality.append(0)
        while self.step_index < self.dataset.n_actions and curr_system_arm not in list(index):
            self.step_index += 1
            curr_system_arm = self.data_dict['action'][self.step_index]
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
        # print(arm)
        arm = arm[self.num_context_dims:]
        reward = self.exp_reward[self.step_index][arm]
        self.reward.append(self.reward[-1] + reward)
        regret = np.max(self.exp_reward[self.step_index]) - reward
        self.regret.append(self.regret[-1] + regret)
        random_arm = np.random.randint(0, len(self.exp_reward[self.step_index]))
        random_regret = np.max(self.exp_reward[self.step_index]) - self.exp_reward[self.step_index][random_arm]
        self.random_regret.append(self.random_regret[-1] + random_regret)
        self.step_index += 1
        return reward
    
    def GetRegrets(self):
        return self.regret
    
    def PlotRegret(self, img_name=None, algo_name='Ensemble Sampling'):
        plt.plot(self.regret, label=algo_name)
        plt.plot(self.random_regret, label='Random Algorithm')
        plt.xlabel('Steps')
        plt.ylabel('Regret')
        plt.title('Regret Plot for Random Algorithm and ' + algo_name)
        plt.legend()
        if img_name is not None:
            plt.savefig(img_name)
        else:
            plt.show()
