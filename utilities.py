from abc import abstractmethod
from abc import ABCMeta as interface

import numpy as np

import scipy as sp
from scipy import linalg
from scipy.spatial import Delaunay as scipydelaunay

from matplotlib import pyplot as plt

import tensorflow as tf
import gpflow


# ---------------------------------------------------------------------------*/
# - synthesize a digital linear-quadratic regulator

def make_dlqr(a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesize a digital linear-quadratic regulator (lqr) using matrices ``a``, ``b``, ``q`` and ``r``,
    where ``a`` and ``b`` are the system and input matrices of a plant to be regulated,
    and where ``q`` and ``r`` are the state and control cost matrices, respectively.
    The method returns a tuple with the regulator matrix K to be used as u = -K*x,
    where x is the full state of the plant; and the solution to an algebraic
    Riccati equation, which was used during the synthesis of K matrix.
    """
    a, b, q, r = map(np.atleast_2d, (a, b, q, r))
    p = sp.linalg.solve_discrete_are(a, b, q, r)

    #                      ~~~~ bpb ~~~~~         ~~ bpa ~~~
    #                     |              |       |          |
    # lqr gain, i.e. k = (b.T * p * b + r)^-1 * (b.T * p * a)
    #                     |     |                |     |
    #                     ~~ bp ~                ~~ bp ~
    bp = b.T.dot(p)
    bpb = bp.dot(b)
    bpb += r
    bpa = bp.dot(a)
    control = np.linalg.solve(bpb, bpa)
    solution = p

    return control, solution


# ---------------------------------------------------------------------------*/
# - delaunay triangulation

class delaunay(metaclass=interface):
    @abstractmethod
    def find_simplex(self, points: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def simplices(self) -> tf.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def nsimplex(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def points(self) -> tf.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def npoints(self) -> int:
        raise NotImplementedError


# ---------------------------------------------------------------------------*/
# - one-dimensional delaunay triangulation

class delaunay_1d(delaunay):
    """
    Class that mimics scipy delaunay algorithm for the triangulation of 1D data
    """
    def __init__(self, points: tf.Tensor) -> None:

        # check dimensions
        if points.shape[1] > 1:
            raise AttributeError('err > this class is designed for 1D inputs')
        if points.shape[0] > 2:
            raise AttributeError('err > this class supports only two points')

        self._points = points
        self._npoints = len(points)

        # there is only one simplex
        self._nsimplex = len(points) - 1

        # limits of this triangulation space
        self._lim_min = tf.reduce_min(points)
        self._lim_max = tf.reduce_max(points)

        # indices of the points forming the simplices in this triangulation
        #
        # shape of simplices is (nsimplex, ndim + 1),
        # see the documentation of scipy delaunay algorithm
        #
        # the following single row represents one simplex formed by two points
        self._simplices = tf.constant([[0, 1]])

    def find_simplex(self, points: tf.Tensor) -> tf.Tensor:
        """
        Locate given ``points`` in the single simplex of this delaunay representation, and
        return either 0 if a point is located, or -1 otherwise.
        """
        points = tf.squeeze(points)

        return tf.where(
            condition=tf.math.logical_or(points > self._lim_max, points < self._lim_min),
            x=-1, y=0)

    @property
    def simplices(self) -> tf.Tensor:
        return self._simplices

    @property
    def nsimplex(self) -> int:
        return self._nsimplex

    @property
    def points(self) -> tf.Tensor:
        return self._points

    @property
    def npoints(self) -> int:
        return self._npoints


# ---------------------------------------------------------------------------*/
# - multi-dimensional delaunay triangulation

class delaunay_nd(delaunay):
    def __init__(self, points: tf.Tensor) -> None:
        self._tri = scipydelaunay(points)

    def find_simplex(self, points: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor(self._tri.find_simplex(points), dtype=tf.int64)

    @property
    def simplices(self) -> tf.Tensor:
        return tf.convert_to_tensor(self._tri.simplices, dtype=tf.int64)

    @property
    def nsimplex(self) -> int:
        return self._tri.nsimplex

    @property
    def points(self) -> tf.Tensor:
        return tf.convert_to_tensor(self._tri.points, dtype=gpflow.default_float())

    @property
    def npoints(self) -> int:
        return self._tri.npoints


# ---------------------------------------------------------------------------*/
# - gaussian process sampler

class gaussianprocess_sampler:
    def __init__(self) -> None:
        self._mean = None
        self._kernel = None
        self._discretization = None
        self._noise_var = None
        self._alphas = None

    def _initialize(self, mean: tf.Tensor, covariance: tf.Tensor, samples_n: int) -> None:
        """
        Initialize sampler that takes the ``samples_n`` number of samples.
        The sampling of a random multivariate normal distrubution is based on ``mean`` and ``covariance``.
        Sampler properties, such as kernel, discretization and observation noise, are expected to be properly set elsewhere.
        """

        # sample normal distribution
        samples = np.random.multivariate_normal(mean, covariance, size=samples_n)

        cho = linalg.cho_factor(covariance, lower=True)
        self._alphas = [
            linalg.cho_solve(cho, samples[[sample_loc], :].T) for sample_loc in range(samples_n)]

    def __call__(self, domain: tf.Tensor, with_noise: bool = False) -> tf.Tensor:
        k = self._kernel.K(domain, self._discretization)
        mean = self._mean(domain)
        noise_var = self._noise_var

        def sample(alpha: tf.Tensor) -> tf.Tensor:
            y = mean + tf.matmul(k, alpha)
            if with_noise:
                y += tf.sqrt(noise_var) * tf.random.normal(tf.shape(y), dtype=gpflow.default_float())
            return y

        return [sample(alpha) for alpha in self._alphas]

    @property
    def samples_n(self) -> int: return len(self._alphas)

# ---------------------------------------------------------------------------*/
# - gaussian process

class gaussianprocess:
    def __init__(
            self,
            mean_fn, cov_fn: gpflow.kernels.Kernel, obsv_ns_var: float,
            dims_n: tuple[int, int]) -> None:

        # define initial observed zero-data at the origin, i.e. x=0, y=0.
        train_i = tf.zeros((1, dims_n[0]), dtype=gpflow.default_float())
        train_o = tf.zeros((1, dims_n[1]), dtype=gpflow.default_float())

        # create a gaussian process regression model
        self.gp = gpflow.models.GPR(
            (train_i, train_o),
            cov_fn, mean_fn, noise_variance=obsv_ns_var)

        self.gp_likelihood_var = obsv_ns_var
        self._update_cache()

    def new_sampler(
            self,
            discretization: tf.Tensor,
            samples_n, noise_var: float) -> gaussianprocess_sampler:

        sampler = gaussianprocess_sampler()
        sampler._mean = self.gp.mean_function
        sampler._kernel = self.gp.kernel
        sampler._discretization = discretization
        sampler._noise_var = noise_var

        # initialize this sampler with the actual mean and covariance matrices; these
        # matrices have extra last dimensions removed before passing
        # to the initialization routine
        mean, cov = self.predict(discretization, full_cov=True)
        mean = tf.squeeze(mean, axis=-1)
        cov = tf.squeeze(cov, axis=-1)
        sampler._initialize(mean, cov, samples_n)

        return sampler

    def predict(self, test: tf.Tensor, full_cov: bool = False) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Predict mean and variance on a test input ``test_i``. If ``full_cov`` flag
        is set to True, this method returns predicts the full covariance.
        Note that this method expands the dimensions of outputs,
        e.g. for ``test_i`` of size (1000, 1), the mean
        will have size (1000, 1), the variance will
        have size (1000, 1), whereas the
        covariance will have size
        (1000, 1000, 1).
        """
        mean = self.gp.mean_function(test)
        cov = self.gp.kernel.K(self.train_i, test)

        a = tf.linalg.triangular_solve(self.cho_predict, cov, lower=True)
        mean = tf.matmul(a, self.alpha, transpose_a=True) + mean

        if full_cov:
            cov = self.gp.kernel.K(test)
            var = cov - tf.matmul(a, a, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(self.train_o)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            cov = self.gp.kernel.K_diag(test)
            var = cov - tf.reduce_sum(tf.square(a), axis=0)
            var = tf.tile(tf.reshape(var, (-1, 1)), [1, tf.shape(self.train_o)[1]])

        return mean, var

    def update_data(self, train_i: tf.Tensor, train_o: tf.Tensor) -> None:
        # combine existing and new training data together
        train_i = tf.concat([self.train_i, train_i], axis=0)
        train_o = tf.concat([self.train_o, train_o], axis=0)

        self.gp.data = train_i, train_o
        self._update_cache()

    def _update_cache(self) -> None:

        # calculate covariances between training inputs corrupted by observation noise,
        # i.e. K(X, X) + sigma^2 * I, see (2.21) from Rasmussen & Williams, 2006
        train_i = self.train_i
        cov = self.gp.kernel.K(train_i) + tf.eye(train_i.shape[0], dtype=gpflow.default_float()) * tf.constant(self.gp_likelihood_var, dtype=gpflow.default_float())

        # training outputs y, or targets
        train_o = self.train_o - self.gp.mean_function(train_i)

        # use cholesky factorization to perform the covariance matrix inversion,
        # see (2.25) from Rasmussen & Williams, 2006
        self.cho_predict = tf.linalg.cholesky(cov)

        # ... and compute the product between the cholesky and the outputs, i.e.
        # alpha = (K + sigma^2 * I)^-1 * y, see (2.25) from Rasmussen & Williams, 2006
        self.alpha = tf.linalg.triangular_solve(self.cho_predict, train_o, lower=True)

    @property
    def train_i(self) -> tf.Tensor: return self.gp.data[0]

    @property
    def train_o(self) -> tf.Tensor: return self.gp.data[1]


# ---------------------------------------------------------------------------*/
# - tensorflow equivalent of numpy ravel_multi_index

def tf_ravel_multi_index(multi_index, dims):
    strides = tf.math.cumprod(dims, exclusive=True, reverse=True)
    return tf.reduce_sum(multi_index * tf.expand_dims(strides, 1), axis=0)


# ---------------------------------------------------------------------------*/
# - plot triangulation in 3D

def plot3d_triangulation(triangulation) -> tuple:
    parameters = triangulation.parameters
    points = triangulation.points
    simplices = triangulation.simplices(np.arange(triangulation.nsimplex))

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    ax.plot_trisurf(
        points[:, 0], points[:, 1], parameters[:, 0],
        triangles=simplices)

    return fig, ax
