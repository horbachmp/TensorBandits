# import numpy as np
from itertools import product
from utils.tensor import *
from utils.bandit import *


class Random():
    def __init__(self, dimensions, bandit, total_steps=20000, img_name=None, update_arm_on_step=None, delete_arm_on_step=None) -> None:
        self.dimensions = dimensions
        self.bandit  = bandit
        self.total_steps = total_steps
        self.img_name = img_name
        self.update_arm_on_step = update_arm_on_step
        self.delete_arm_on_step = delete_arm_on_step
        self.all_arms = set(product(*list(map(lambda x: list(range(x)), self.dimensions))))

    def Step(self, arm):
        reward = self.bandit.PlayArm(arm)

    
    def UpdateArms(self, dim, ind, delta):
        combinations = set(product(*list(map(lambda x: list(range(x)), self.dimensions))))
        filtered_combinations = [combo for combo in combinations if combo[dim] == ind]
        for arm in filtered_combinations:
            self.bandit.UpdateArm(arm, delta)

    def DeleteArms(self, dim, ind):
        combinations = set(product(*list(map(lambda x: list(range(x)), self.dimensions))))
        filtered_combinations = [combo for combo in combinations if combo[dim] == ind]
        for arm in filtered_combinations:
            if arm in self.all_arms:
                self.all_arms.remove(arm)
    
    def PlayAlgo(self):
        updated = False
        deleted = False
        for step in range(0, self.total_steps):
            if self.update_arm_on_step is not None and not updated and step >= self.update_arm_on_step:
                updated = True
                self.UpdateArms(2, 2, 0.6)
            if (self.delete_arm_on_step is not None) and (not deleted) and step >= self.delete_arm_on_step:
                deleted = True
                self.DeleteArms(0, 2)
            arm = np.random.randint(0, high=self.dimensions, size=len(self.dimensions))
            while tuple(arm) not in self.all_arms:
                arm = np.random.randint(0, high=self.dimensions, size=len(self.dimensions))
            self.Step(arm)

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
    algo = Random(dimensions=[3,3,3], bandit=bandit, img_name="random_upd4000", update_arm_on_step=4000)
    algo.PlayAlgo()

if __name__ == "__main__":
    main()