import numpy as np
import matplotlib.pyplot as plt
import time

from utils.bandit import TensorBandit
from tensor_train_algo import TensorTrainAlgo
from elimination import TensorElimination
from epoch_greedy import TensorEpochGreedy
from ensemble_sampling import EnsembleSampling

def errorfill(x, y, yerr, label=None, color=None, alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.get_next_color()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color, label=label)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


def main():
    # dimensions = [10, 10, 10]
    dimensions = [5, 5, 5]
    cdims=1
    X = np.random.rand(*dimensions)

    real_best = np.unravel_index(np.argmax(X[0]), X[0].shape)
    print("real best arm for context 0:",  real_best)
    print(X[0][real_best])

    explore_steps = 2000
    total_steps = 5000

    regrets_tt_runs = []
    regrets_elim_runs = []
    regrets_ep_runs = []
    regrets_ens_runs = []

    times_tt = []
    times_elim = []
    times_ep = []
    times_ens = []

    for run in range(1):

        print("run:", run)

        print("Running TensorTrainAlgo")
        bandit = TensorBandit(X, 0.5)
        algo_tt = TensorTrainAlgo(dimensions=dimensions, ranks=[1,2,2,1], num_context_dims=cdims, bandit=bandit, explore_steps=explore_steps, total_steps=total_steps)
        start_time = time.time()
        algo_tt.PlayAlgo()
        algo_tt_time = time.time() - start_time
        regrets_tt = bandit.GetRegrets()
        times_tt.append(algo_tt_time)
        regrets_tt_runs.append(regrets_tt)
        print("TensorTrainAlgo finished")



        print("Running TensorElimination")
        bandit = TensorBandit(X, 0.5)
        algo_elim = TensorElimination(dimensions=dimensions, ranks=[2,2,2], num_context_dims=cdims, bandit=bandit, explore_steps=explore_steps, total_steps=total_steps)
        start_time = time.time()
        algo_elim.PlayAlgo()
        algo_elim_time = time.time() - start_time
        regrets_elim = bandit.GetRegrets()
        times_elim.append(algo_elim_time)
        regrets_elim_runs.append(regrets_elim)
        print("TensorElimination finished")

        print("Running EpochGreedy")
        bandit = TensorBandit(X, 0.5)
        algo_ep = TensorEpochGreedy(dimensions=dimensions, ranks=[2,2,2], num_context_dims=cdims, bandit=bandit, explore_steps=explore_steps, total_steps=total_steps)
        start_time = time.time()
        algo_ep.PlayAlgo()
        algo_ep_time = time.time() - start_time
        regrets_ep = bandit.GetRegrets()
        times_ep.append(algo_ep_time)
        regrets_ep_runs.append(regrets_ep)
        print("EpochGreedy finished")

        print("Running Ensemble Sampling")
        bandit = TensorBandit(X, 0.5)
        ranks=[2,2,2]
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
        algo = EnsembleSampling(dimensions=dimensions, ranks=ranks, bandit=bandit, num_context_dims=1, prior_mus=mus, prior_sigmas=sigmas, perturb_noise=0.1, explore_steps=explore_steps, total_steps=total_steps)
        start_time = time.time()
        algo.PlayAlgo()
        algo_ens_time = time.time() - start_time
        regrets_ens = bandit.GetRegrets()
        times_ens.append(algo_ens_time)
        regrets_ens_runs.append(regrets_ens)
        print("Ensemble Sampling finished")

    mean_regrets_tt = np.mean(regrets_tt_runs, axis=0)
    std_regrets_tt = np.std(regrets_tt_runs, axis=0)

    mean_regrets_elim = np.mean(regrets_elim_runs, axis=0)
    std_regrets_elim = np.std(regrets_elim_runs, axis=0)

    mean_regrets_ep = np.mean(regrets_ep_runs, axis=0)
    std_regrets_ep = np.std(regrets_ep_runs, axis=0)

    mean_regrets_ens = np.mean(regrets_ens_runs, axis=0)
    std_regrets_ens = np.std(regrets_ens_runs, axis=0)

    plt.figure(figsize=(10, 6))

    # plt.errorbar(range(len(mean_regrets_tt)), mean_regrets_tt, yerr=std_regrets_tt, alpha=0.3, label=f'TensorTrainAlgo (time: {np.mean(times_tt):.2f} sec)', markeredgecolor='none')
    # plt.errorbar(range(len(mean_regrets_elim)), mean_regrets_elim, yerr=std_regrets_elim, label=f'TensorElimination (time: {np.mean(times_elim):.2f} sec)', alpha=0.5)
    # plt.errorbar(range(len(mean_regrets_ep)), mean_regrets_ep, yerr=std_regrets_ep, label=f'TensorEpochGreedy (time: {np.mean(times_ep):.2f} sec)', alpha=0.5)
    # plt.errorbar(range(len(mean_regrets_ucb)), mean_regrets_ucb, yerr=std_regrets_ucb, label=f'Vect_UCB_1 (time: {np.mean(times_ucb):.2f} sec)', alpha=0.5, markeredgecolor='none')

    
    errorfill(range(len(mean_regrets_ep)), mean_regrets_ep, yerr=std_regrets_ep, label=f'TensorEpochGreedy (time: {np.mean(times_ep):.2f} sec)')
    errorfill(range(len(mean_regrets_elim)), mean_regrets_elim, yerr=std_regrets_elim, label=f'TensorElimination (time: {np.mean(times_elim):.2f} sec)')
    errorfill(range(len(mean_regrets_ens)), mean_regrets_ens, yerr=std_regrets_ens, label=f'Ensemble Sampling (time: {np.mean(times_ens):.2f} sec)')
    errorfill(range(len(mean_regrets_tt)), mean_regrets_tt, yerr=std_regrets_tt, label=f'TensorTrainAlgo (time: {np.mean(times_tt):.2f} sec)')

    
    plt.xlabel('Iteration')
    plt.ylabel('Regret')
    plt.title('Regrets Comparison')
    plt.legend()

    plt.grid(True)

    plt.savefig('/home/maryna/HSE/Bandits/TensorBandits/context_algs/compare_pictures/first_try.png')
    plt.show()

if __name__ == "__main__":
    main()