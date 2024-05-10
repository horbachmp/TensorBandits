import numpy as np
import matplotlib.pyplot as plt
import time

from utils.bandit import TensorBandit
from tensor_train_algo import TensorTrainAlgo
from elimination import TensorElimination
from epoch_greedy import TensorEpochGreedy
from ucb import Vect_UCB_1 



def main():
    seed = 42
    dimensions = [10, 10, 10]
    X = np.random.rand(*dimensions)

    real_best = np.unravel_index(np.argmax(X), X.shape)
    print("real best arm:",  real_best)
    print(X[real_best])

    explore_steps = 2000
    total_steps = 5000

    print("Running TensorTrainAlgo")
    bandit = TensorBandit(X, 0.5)
    algo_tt = TensorTrainAlgo(dimensions=dimensions, ranks=[1,2,2,1], bandit=bandit, explore_steps=explore_steps, total_steps=total_steps)
    start_time = time.time()
    algo_tt.PlayAlgo()
    algo_tt_time = time.time() - start_time
    regrets_tt = bandit.GetRegrets()
    print("TensorTrainAlgo finished")


    print("Running TensorElimination")
    bandit = TensorBandit(X, 0.5)
    algo_elim = TensorElimination(dimensions=dimensions, ranks=[2,2,2], bandit=bandit, explore_steps=explore_steps, total_steps=total_steps)
    start_time = time.time()
    algo_elim.PlayAlgo()
    algo_elim_time = time.time() - start_time
    regrets_elim = bandit.GetRegrets()
    print("TensorElimination finished")

    print("Running EpochGreedy")
    bandit = TensorBandit(X, 0.5)
    algo_ep = TensorEpochGreedy(dimensions=dimensions, ranks=[2,2,2], bandit=bandit, explore_steps=explore_steps, total_steps=total_steps)
    start_time = time.time()
    algo_ep.PlayAlgo()
    algo_ep_time = time.time() - start_time
    regrets_ep = bandit.GetRegrets()
    print("EpochGreedy finished")

    print("Running UCB")
    bandit = TensorBandit(X, 0.5)
    algo_ucb = Vect_UCB_1(dimensions=dimensions, bandit=bandit, explore_steps=explore_steps, total_steps=total_steps)
    start_time = time.time()
    algo_ucb.PlayAlgo()
    algo_ucb_time = time.time() - start_time
    regrets_ucb = bandit.GetRegrets()
    print("UCB finished")

    

    plt.figure(figsize=(10, 6))

    plt.plot(regrets_tt, label=f'TensorTrainAlgo (time: {algo_tt_time:.2f} sec)')
    plt.plot(regrets_elim, label=f'TensorElimination (time: {algo_elim_time:.2f} sec)')
    plt.plot(regrets_ep, label=f'TensorEpochGreedy (time: {algo_ep_time:.2f} sec)')
    plt.plot(regrets_ucb, label=f'Vect_UCB_1 (time: {algo_ucb_time:.2f} sec)')

    plt.xlabel('Iteration')
    plt.ylabel('Regret')
    plt.title('Regrets Comparison')
    plt.legend()

    plt.grid(True)

    plt.savefig('../compare_pictures/regrets_comparison_d_10_10_10_r_2.png')
    plt.show()

if __name__ == "__main__":
    main()