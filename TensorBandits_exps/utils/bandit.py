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
