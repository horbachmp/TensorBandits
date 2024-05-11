from utils.tensor import *
from tensorly.decomposition import tucker
import numpy as np
import matplotlib.pyplot as plt
import random

class TensorBandit:
    def __init__(self, reward_tesor, noise_level, log=False) -> None:
        self.dimensions = reward_tesor.shape
        self.X = reward_tesor
        self.noise_level = noise_level
        self.log=log
        
        self.opt_arm_rew = np.max(self.X)
        self.regrets = [0]
        self.random_regrets = [0]

    def UpdateArm(self, index, delta):
        arm_tensor = np.zeros(self.dimensions, dtype=int)
        arm_tensor[tuple(index)] = delta
        self.X += arm_tensor

    def PlayArm(self, index, context):
        arm_tensor = np.zeros(self.dimensions, dtype=int)
        arm_tensor[tuple(index)] = 1
        determ_reward = np.sum(self.X * arm_tensor)
        noise = np.random.normal(0, self.noise_level, 1)
        reward = determ_reward + noise[0]
        if self.log:
            print(index)
            print(determ_reward, noise[0])
            print(reward)
        opt_arm_for_context = np.max(self.X[context])
        regret = opt_arm_for_context - reward
        self.regrets.append(self.regrets[-1] + regret)
        tens = self.X[context][0]
        random_arm = np.array([random.randint(0, dim - 1) for dim in tens.shape])
        random_regret = opt_arm_for_context - tens[tuple(random_arm)]
        self.random_regrets.append(self.random_regrets[-1]+random_regret)
        return np.array([reward])
    
    
    # def PlotRegret(self, img_name=None):
    #     plt.plot(self.regrets, label='Ensemble Sampling')
    #     plt.plot(self.random_regrets, label='Random Algorithm')
    #     plt.xlabel('Steps')
    #     plt.ylabel('Regret')
    #     plt.title('Regret Plot for Ensemble Sampling and Random Algorithm')
    #     plt.legend()
    #     if img_name is not None:
    #         plt.savefig(img_name)
    #     else:
    #         plt.show()


    def PlotRegret(self, img_name=None):
        plt.plot(self.regrets, label='Ensemble Sampling')
        # plt.plot(self.random_regrets, label='Random Algorithm')
        plt.xlabel('Steps')
        plt.ylabel('Regret')
        plt.title('Regret Plot for Ensemble Sampling')
        # plt.legend()
        if img_name is not None:
            plt.savefig(img_name)
        else:
            plt.show()
