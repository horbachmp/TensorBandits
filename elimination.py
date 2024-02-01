import numpy as np
from tensorly.decomposition import tucker
import tensorly as tl
import math
from scipy.linalg import null_space
from itertools import product

def prod(arg):
    """ returns the product of elements in arg.
    arg can be list, tuple, set, and array with numerical values. """
    ret = 1
    for i in range(0, len(arg)):
        ret = ret * arg[i]
    return ret

class Tensor(object):
    """
    Store a basic Tensor.
    """

    def __init__(self, data=None, shape=None):
        """
        Constructor for Tensor object.
        ----------
        :param data: can be numpy, array or list.
        :param shape: can be tuple, numpy.array, or list of integers
        :return: constructed Tensor object.
        ----------
        """

        if data is None:
            raise ValueError('Tensor: first argument cannot be empty.')

        if data.__class__ == list or data.__class__ == np.ndarray:
            if data.__class__ == list:
                data = np.array(data)
        else:
            print(data.__class__)
            raise ValueError('Tensor: first argument should be either list or numpy.ndarray')

        if shape:
            if shape.__class__ != tuple:
                if shape.__class__ == list:
                    if len(shape) < 2:
                        raise ValueError('Tensor: second argument must be a row vector with at least two elements.')
                    elif shape[0].__class__ != int:
                        if len(shape[0]) == 1:
                            shape = [y for x in shape for y in x]
                        else:
                            raise ValueError('Tensor: second argument must be a row vector with integers.')
                    else:
                        shape = tuple(shape)
                elif shape.__class__ == np.ndarray:
                    if shape.ndim != 2:
                        raise ValueError('Tensor: second argument must be a row vector with integers.')
                    elif shape[0].__class__ != np.int64:
                        if len(shape[0]) == 1:
                            shape = [y for x in shape for y in x]
                        else:
                            raise ValueError('Tensor: second argument must be a row vector with integers.')
                    else:
                        shape = tuple(shape)
                else:
                    raise ValueError('Tensor: second argument must be a row vector (tuple).')
            else:
                if len(shape) < 2:
                    raise ValueError('Tensor: second argument must be a row vector with at least two elements.')
                elif len(shape[0]) != 1:
                    raise ValueError('Tensor: second argument must be a row vector.')
            if prod(shape) != data.size:
                raise ValueError("Tensor: size of data does not match specified size of Tensor.")
        else:
            shape = tuple(data.shape)

        self.shape = shape
        self.data = data.reshape(shape, order='F')
        self.ndims = len(self.shape)
        print(self.data)
        print(self.shape)
        print(self.ndims)

    def __str__(self):
        string = "Tensor of size {0} with {1} elements.\n".format(self.shape, prod(self.shape))
        return string

    def size(self):
        """ Returns the number of elements in the Tensor """
        return self.data.size

    def copy(self):
        """ Returns a deepcpoy of Tensor object """
        return Tensor(self.data)

    def dimsize(self, idx=None):
        """ Returns the size of the specified dimension """
        if idx is None:
            raise ValueError('Please specify the index of that dimension.')
        if idx.__class__ != int:
            raise ValueError('Index of the dimension must be an integer.')
        if idx >= self.ndims:
            raise ValueError('Index exceeds the number of dimensions.')
        return self.shape[idx]

    def permute(self, order=None):
        """ Returns a Tensor permuted by the order specified."""
        if order is None:
            raise ValueError("Permute: Order must be specified.")

        if order.__class__ == list or order.__class__ == tuple:
            order = np.array(order)

        if self.ndims != len(order):
            raise ValueError("Permute: Invalid permutation order.")

        if not (sorted(order) == np.arange(self.ndims)).all():
            raise ValueError("Permute: Invalid permutation order.")

        newdata = self.data.copy()

        newdata = newdata.transpose(order)

        return Tensor(newdata)

    def ipermute(self, order=None):
        """ Returns a Tensor permuted by the inverse of the order specified """
        if order is None:
            raise ValueError('Ipermute: please specify the order.')

        if order.__class__ == np.array or order.__class__ == tuple:
            order = list(order)
        else:
            if order.__class__ != list:
                raise ValueError('Ipermute: permutation order must be a list.')

        if not self.ndims == len(order):
            raise ValueError("Ipermute: invalid permutation order.")
        if not ((sorted(order) == np.arange(self.ndims)).all()):
            raise ValueError("Ipermute: invalid permutation order.")

        iorder = [order.index(idx) for idx in range(0, len(order))]

        return self.permute(iorder)

    def tondarray(self):
        """ Returns data of the Tensor with a numpy.ndarray object """
        return self.data

    def ttm(self, mat=None, mode=None, option=None):
        """ Multiplies the Tensor with the given matrix.
            the given matrix is a single 2-D array with list or numpy.array."""
        if mat is None:
            raise ValueError('Tensor/TTM: matrix (mat) needs to be specified.')

        if mode is None or mode.__class__ != int or mode > self.ndims or mode < 1:
            raise ValueError('Tensor/TTM: mode must be between 1 and NDIMS(Tensor).')

        if mat.__class__ == list:
            matrix = np.array(mat)
        elif mat.__class__ == np.ndarray:
            matrix = mat
        else:
            raise ValueError('Tensor/TTM: matrix must be a list or a numpy.ndarray.')

        if len(matrix.shape) != 2:
            raise ValueError('Tensor/TTM: first argument must be a matrix.')

        if matrix.shape[1] != self.shape[mode - 1]:
            raise ValueError('Tensor/TTM: matrix dimensions must agree.')

        dim = mode - 1
        n = self.ndims
        shape = list(self.shape)
        order = [dim] + range(0, dim) + range(dim + 1, n)
        new_data = self.permute(order).data
        new_data = new_data.reshape(shape[dim], prod(shape) / shape[dim])
        if option is None:
            new_data = np.dot(matrix, new_data)
            p = matrix.shape[0]
        elif option == 't':
            new_data = np.dot(matrix.transpose(), new_data)
            p = matrix.shape[1]
        else:
            raise ValueError('Tensor/TTM: unknown option')
        new_shape = [p] + shape[0:dim] + shape[dim + 1:n]
        new_data = Tensor(new_data.reshape(new_shape))
        new_data = new_data.ipermute(order)

        return new_data

    def norm(self):
        """Return the Frobenius norm of the Tensor"""
        return np.linalg.norm(self.data)

    def unfold(self, n=None):
        """Return the mode-n unfold of the Tensor."""
        if n is None:
            raise ValueError('Tensor/UNFOLD: unfold mode n (int) needs to be specified.')
        N = self.ndims
        temp1 = [n]
        temp2 = range(n)
        temp3 = range(n + 1, N)
        temp1[len(temp1):len(temp1)] = temp2
        temp1[len(temp1):len(temp1)] = temp3
        xn = self.permute(temp1)
        xn = xn.tondarray()
        xn = xn.reshape([xn.shape[0], np.prod(xn.shape) / xn.shape[0]])
        return xn

    def nvecs(self, n=None, r=None):
        """Return first r eigenvectors of the mode-n unfolding matrix"""
        if n is None:
            raise ValueError('Tensor/NVECS: unfold mode n (int) needs to be specified.')
        if r is None:
            raise ValueError('Tensor/NVECS: the number of eigenvectors r needs to be specified.')
        xn = self.unfold(n)
        [eigen_value, eigen_vector] = np.linalg.eig(xn.dot(xn.transpose()))
        return eigen_vector[:, range(r)]



class Tenmat(object):
    """
    Store a Matricization of a Tensor object.
    """

    def __init__(self, x=None, rdim=None, cdim=None, tsize=None):
        """
        Create a Tenmat object from a given Tensor X
         ----------
        :param x: dense Tensor object.
        :param rdim: an one-dim array representing the arranged dimension index for the matrix column
        :param cdim: an one-dim array representing the arranged dimension index for the matrix row
        :param tsize: a tuple denoting the size of the original tensor
        :return: constructed Matricization of a Tensor object.
        ----------
        """

        if x is None:
            raise ValueError('Tenmat: first argument cannot be empty.')

        if x.__class__ == Tensor:
            # convert a Tensor to a matrix
            if rdim is None:
                raise ValueError('Tenmat: second argument cannot be empty.')

            if rdim.__class__ == list or rdim.__class__ == int:
                rdim = np.array(rdim) - 1

            self.shape = x.shape

            ##################
            if cdim is None:
                cdim = np.array([y for y in range(0, x.ndims) if y not in np.zeros(x.ndims - 1) + rdim])
            elif cdim.__class__ == list or cdim.__class__ == int:
                cdim = np.array(cdim) - 1
            else:
                raise ValueError("Tenmat: incorrect specification of dimensions.")
            ##############
            tmp = sorted(np.append(rdim, cdim))
            if not (list(range(0, x.ndims)) == tmp):
                raise ValueError("Tenmat: second argument must be a list or an integer.")

            self.rowIndices = rdim
            self.colIndices = cdim

            x = x.permute(np.append(rdim, cdim))

            ##################
            if type(rdim) != np.ndarray:
                row = prod([self.shape[y] for y in [rdim]])
            else:
                row = prod([self.shape[y] for y in rdim])

            if type(cdim) != np.ndarray:
                col = prod([self.shape[y] for y in [cdim]])
            else:
                col = prod([self.shape[y] for y in cdim])
            ##################

            self.data = x.data.reshape([row, col], order='F')
        elif x.__class__ == np.ndarray:
            # copy a matrix to a Tenmat object
            if len(x.shape) != 2:
                raise ValueError("Tenmat: first argument must be a 2-D numpy array when converting a matrix to Tenmat.")

            if tsize is None:
                raise ValueError("Tenmat: Tensor size must be specified as a tuple.")
            else:
                if rdim is None or cdim is None or rdim.__class__ != list or cdim.__class__ != list:
                    raise ValueError("Tenmat: second and third arguments must be specified with list.")
                else:
                    rdim = np.array(rdim) - 1
                    cdim = np.array(cdim) - 1
                    if prod([tsize[idx] for idx in rdim]) != x.shape[0]:
                        raise ValueError("Tenmat: matrix size[0] does not match the Tensor size specified.")
                    if prod([tsize[idx] for idx in cdim]) != x.shape[1]:
                        raise ValueError("Tenmat: matrix size[1] does not match the Tensor size specified.")
            self.data = x
            self.rowIndices = rdim
            self.colIndices = cdim
            self.shape = tsize

    def copy(self):
        # returns a deepcpoy of Tenmat object
        return Tenmat(self.data, self.rowIndices, self.colIndices, self.shape)

    def totensor(self):
        # returns a Tensor object based on a Tenmat
        order = np.append(self.rowIndices, self.colIndices)
        data = self.data.reshape([self.shape[idx] for idx in order], order='F')
        t_data = Tensor(data).ipermute(list(order))
        return t_data

    def tondarray(self):
        # returns data of a Tenmat with a numpy.ndarray object
        return self.data

    def __str__(self):
        ret = ""
        ret += "Matrix corresponding to a Tensor of size {0}\n".format(self.shape)
        ret += "Row Indices {0}\n".format(self.rowIndices + 1)
        ret += "Column Indices {0}\n".format(self.colIndices + 1)
        return ret


def pro_to_trace_norm(z, tau):
    m = z.shape[0]
    n = z.shape[1]
    if 2 * m < n:
        [U, Sigma2, V] = np.linalg.svd(np.dot(z, z.T))
        S = np.sqrt(Sigma2)
        tol = np.max(z.shape) * (2 ** int(math.log(max(S), 2))) * 2.2204 * 1E-16
        k = np.sum(S > max(tol, tau))
        mid = [max(S[i] - tau, 0) * 1.0 / S[i] for i in range(k)]
        X = np.dot(np.dot(U[:, 0:k], np.dot(np.diag(mid), U[:, 0:k].T)), z)
        return X, k, Sigma2
    if m > 2 * n:
        z = z.T
        [U, Sigma2, V] = np.linalg.svd(np.dot(z, z.T))
        S = np.sqrt(Sigma2)
        tol = np.max(z.shape) * (2 ** int(math.log(max(S), 2))) * 2.2204 * 1E-16
        k = np.sum(S > max(tol, tau))
        mid = [max(S[i] - tau, 0) * 1.0 / S[i] for i in range(k)]
        X = np.dot(np.dot(U[:, 0:k], np.dot(np.diag(mid), U[:, 0:k].T)), z)
        return X.T, k, Sigma2

    [U, S, V] = np.linalg.svd(z)
    Sigma2 = S ** 2
    k = sum(S > tau)
    X = np.dot(U[:, 0:k], np.dot(np.diag(S[0:k] - tau), V[0:k, :]))
    return X, n, Sigma2


def silrtc(x, omega=None, alpha=None, gamma=None, max_iter=100, epsilon=1e-5, printitn=100):
    """
    Simple Low Rank Tensor Completion (SiLRTC).
    Reference: "Tensor Completion for Estimating Missing Values in Visual Data", PAMI, 2012.
    """

    T = x.data.copy()
    N = x.ndims
    # dim = x.shape
    if printitn == 0:
        printitn = max_iter
    if omega is None:
        omega = x.data * 0 + 1

    if alpha is None:
        alpha = np.ones([N])
        alpha = alpha / sum(alpha)

    if gamma is None:
        gamma = 0.1 * np.ones([N])

    normX = x.norm()
    # initialization
    x.data[omega == 0] = np.mean(x.data[omega == 1])
    errList = np.zeros([max_iter, 1])

    M = list(range(N))
    gammasum = sum(gamma)
    tau = alpha / gamma

    for k in range(max_iter):
        if (k + 1) % printitn == 0 and k != 0 and printitn != max_iter:
            print('SiLRTC: iterations = {0}   difference = {1}\n'.format(k, errList[k - 1]))

        Xsum = 0
        for i in range(N):
            temp = Tenmat(x, i + 1)
            [temp1, tempn, tempSigma2] = pro_to_trace_norm(temp.data, tau[i])
            temp.data = temp1
            M[i] = temp.totensor().data
            Xsum = Xsum + gamma[i] * M[i]

        Xlast = x.data.copy()
        Xlast = Tensor(Xlast)

        x.data = Xsum / gammasum
        x.data = T * omega + x.data * (1 - omega)
        diff = x.data - Xlast.data
        errList[k] = np.linalg.norm(diff) / normX
        if errList[k] < epsilon:
            errList = errList[0:(k + 1)]
            break

    print('SiLRTC ends: total iterations = {0}   difference = {1}\n\n'.format(k + 1, errList[k]))
    return x



class Bandit:
    def __init__(self, dimensions=[2,2]) -> None:
        self.dimensions = np.array(dimensions)
        self.X = np.random.uniform(0, 1, self.dimensions)


    def PlayArm(self, index):
        arm_tensor = np.zeros(self.dimensions, dtype=int)
        arm_tensor[tuple(index)] = 1
        determ_reward = np.sum(self.X * arm_tensor)
        noise = np.random.normal(np.sum(index), 1, 1)
        return determ_reward + noise


# def margin_mult(A, B, dim_ind):
#     tensor_shape = list(A.shape)
#     tensor_shape[dim_ind] = B.shape[dim_ind]
#     print(tensor_shape)
#     result_tensor = np.zeros(tensor_shape)
#     for i1 in range(B.shape[dim_ind]):
#         selected_slice_A = np.take(A, indices=[...], axis=dim_ind)
#         sum_elem = 0
#         for j in range(A.shape[dim_ind]):
#             sum_elem += selected_slice_A[j] * B[i1, j]
#             print(sum_elem)


#     return result_tensor
    

def marginal_multiplication(X, Y, axis):
    result = np.tensordot(X, Y, axes=([axis], [1]))
    result = np.moveaxis(result, len(result.shape) - 1, axis)
    return result
    
# def marginal_multiplication(X, Y, k):
#     X_new = np.moveaxis(X, k, 0)
#     result =  np.einsum('i...,ij->j...', X_new, Y)
#     result = np.moveaxis(result, 0, k)
#     return result


class TensorElimination:
    def __init__(self, dimensions=[4,4], total_steps=10, explore_steps=5, lambda1=1, lambda2=1, conf_int_len=0.5, ranks=[2,2]) -> None:
        self.bandit  = Bandit(dimensions)
        self.dimensions = dimensions
        self.total_steps = total_steps
        self.explore_steps = explore_steps
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.conf_int_len = conf_int_len
        self.ranks = ranks
        self.Reward_vec_est = np.zeros(dimensions)
        self.have_info = np.zeros(dimensions).astype(bool)
        self.all_arms = set(product(*list(map(lambda x: list(range(x)), self.dimensions))))

    def ExploreStep(self, arm):
        reward = self.bandit.PlayArm(arm)
        print(reward)
        arm_tensor = np.zeros(self.dimensions, dtype=int)
        arm_tensor[tuple(arm)] = 1
        self.Reward_vec_est += arm_tensor * reward
        self.have_info = self.have_info | arm_tensor


    def ExploitStep(self, arm):
        reward = self.bandit.PlayArm(arm)

    


    def PlayAlgo(self):
        #exploration
        for step in range(self.explore_steps):
            arm = np.random.randint(0, high=self.dimensions, size=len(self.dimensions))
            print(arm)
            self.ExploreStep(arm)
            print(self.Reward_vec_est)
        # print(np.prod(self.dimensions), self.explore_steps)
        Rew_vec_ini = (np.prod(self.dimensions) / self.explore_steps) * self.Reward_vec_est
        Rew_vec_completed = silrtc(Tensor(Rew_vec_ini), omega=self.have_info)
        Rew_vec_completed = Rew_vec_completed.data
        print(Rew_vec_completed)
        core, factors = tucker(Rew_vec_completed, rank=self.ranks)
        perp_factors = []
        X_res = core
        ind = 0
        for factor in factors:
            perp_factor = null_space(factor.T)
            perp_factors.append(perp_factor)
            print(factor.shape)
            print("X_res", X_res.shape)
            print(X_res)
            X_res = marginal_multiplication(X_res, factor, ind)
            ind += 1
        # print(perp_factors)
        q = np.prod(self.dimensions) - np.prod(np.array(self.dimensions) - np.array(self.ranks))
        print(q)
        print(X_res)
        print(Rew_vec_completed)
        print(np.all(np.isclose(X_res, Rew_vec_completed, atol=1e-5)))
        #reduction

        
        # print(self.all_arms)
        # for k in np.log2(self.total_steps):






def main():
    algo = TensorElimination()
    algo.PlayAlgo()
    # tensor = tl.tensor([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #                     [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
    #                     [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
    #                     [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
    #                     [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
    #                     [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
    #                     [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
    #                     [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
    #                     [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
    #                     [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
    #                     [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
    #                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
    # # core, factors = tucker(tensor, rank=[2, 3])
    # # print(core)

    # ten_arr = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #                     [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
    #                     [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
    #                     [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
    #                     [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
    #                     [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
    #                     [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
    #                     [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
    #                     [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
    #                     [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
    #                     [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
    #                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
    # ten = Tensor(ten_arr)
    # omega = np.zeres_like(ten_arr)
    # ans = silrtc(ten, omega=omega)
    # print(ans.data)










main()