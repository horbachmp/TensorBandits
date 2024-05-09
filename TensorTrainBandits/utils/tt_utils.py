import tensorly
import numpy as np
import numpy as np
from scipy import linalg
from copy import deepcopy
from tensorly.decomposition import tensor_train


def get_tensor_from_tt(cores):
    shape_list = []
    for core in cores:
      shape_list.append(core.shape[1])
    result_tensor = np.ones(shape_list)
    for index, value in np.ndenumerate(result_tensor):
        res = cores[0][:,index[0], :]
        for i, ind in enumerate(index[1:]):
            res = res @ cores[i+1][:, ind,:]
        result_tensor[index] = res
    return result_tensor




def tt_orth(cores):
    for i in range(len(cores) - 1, 0, -1):
        G_i = cores[i].reshape(cores[i].shape[0], -1)
        R, Q = linalg.rq(G_i, mode='economic')
        
        cores[i] = Q.reshape(cores[i].shape)

        G_i1 = cores[i-1].reshape(-1, cores[i-1].shape[-1])
        G_i1 = G_i1 @ R
        cores[i-1] = G_i1.reshape(cores[i-1].shape)
    return cores





def top_k_rows(matrix, k):
    norms = np.linalg.norm(matrix, axis=1)
    sorted_indices = np.argsort(norms)
    return sorted_indices[-k:][::-1]


def optima_tt_max(cores, k=2, rank=2):
    need_restructure = False
    for core in cores:
        if max(core.shape) > core.shape[1]:
          need_restructure = True
          break
    new_cores = deepcopy(cores)
    if need_restructure:
        t = get_tensor_from_tt(cores)
        new_cores = tensor_train(t, rank=rank)
    new_cores = tt_orth(new_cores)
    Q = cores[0][0, :,:]
    N0 = cores[0].shape[1]
    I = np.array([range(N0)]).T
    ind = top_k_rows(Q, k)
    Q = Q[ind, :]
    I = I[ind,:]
    for i in range(1, len(cores)):
        Ni = cores[i].shape[1]
        Gi = cores[i].reshape((cores[i].shape[0], -1))
        Q = Q @ Gi
        Q = Q.reshape((-1, cores[i].shape[-1]))
        I_old = np.kron(I, np.ones((Ni, 1)))
        I_curr = np.kron(np.ones((k, 1)), np.array([range(Ni)]).T)
        I = np.concatenate((I_old, I_curr), axis = 1)
        ind = top_k_rows(Q, k)
        Q = Q[ind, :]
        I = I[ind,:]
    return I[0,:].astype(int)


def tt_sum(cores1, cores2):
    new_cores = []
    core0 = np.concatenate((cores1[0], cores2[0]), axis=2)
    new_cores.append(core0)
    for i in range(1, len(cores1) - 1):
        core_f = np.concatenate((cores1[i], np.zeros((cores1[i].shape[0],cores1[i].shape[1],cores2[i].shape[2]))), axis=2)
        core_s = np.concatenate((np.zeros((cores2[i].shape[0],cores2[i].shape[1],cores1[i].shape[2])), cores2[i]), axis=2)
        core = np.concatenate((core_f, core_s), axis=0)
        new_cores.append(core)

    
    core_last = np.concatenate((cores1[-1], cores2[-1]), axis=0)
    new_cores.append(core_last)
    return new_cores


def tt_prod(cores1, cores2):
    new_cores = []
    for i in range(len(cores1)):
        n = cores1[i].shape[1]
        core_parts = []
        for j in range(n):
            part = np.kron(cores1[i][:, j, :], cores2[i][:, j, :])[:, np.newaxis, :]
            core_parts.append(part)
        core = np.concatenate(core_parts, axis=1)
        new_cores.append(core)
    return new_cores
