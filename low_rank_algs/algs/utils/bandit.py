from utils.tensor import *

from tensorly.decomposition import tucker
import numpy as np
import matplotlib.pyplot as plt

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
        
        random_arm = np.random.randint(0, high=self.dimensions, size=len(self.dimensions))
        random_reward = self.X[tuple(random_arm)]
        self.random_regrets.append(self.random_regrets[-1] + (self.opt_arm_rew - random_reward))
        return np.array([reward])
    
    def PlotRegret(self, img_name=None, algo_name="Tensor Elimination", plot_random=True):
        plt.plot(self.regrets, label=algo_name)
        if plot_random:
            plt.plot(self.random_regrets, label='Random Algorithm')
            plt.legend()
        plt.xlabel('Steps')
        plt.ylabel('Regret')
        plt.title('Regret Plot for ' + algo_name)
        
        if img_name is not None:
            plt.savefig(img_name)
        else:
            plt.show()
    
    def GetRegrets(self):
        return self.regrets
