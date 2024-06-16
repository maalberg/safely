import numpy as np
import scipy as sp
from scipy import linalg
import tensorflow as tf
import gpflow
from matplotlib import pyplot as plt


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
# - one-dimensional delaunay triangulation

class delaunay1d:
    """
    Class that mimics scipy delaunay algorithm for the triangulation of 1D data
    """
    def __init__(self, points: np.ndarray) -> None:
        if points.shape[1] > 1:
            raise AttributeError('err > this class is designed for 1D inputs')
        if points.shape[0] > 2:
            raise AttributeError('err > this class supports only two points')

        self.points = points

        # there will be only one simplex
        self.nsimplex = len(points) - 1

        self._bound_min = np.min(points)
        self._bound_max = np.max(points)

        # indices of the points forming the simplices in this triangulation
        #
        # shape of simplices is (nsimplex, ndim + 1),
        # see the documentation of scipy delaunay algorithm
        #
        # the following single row represents one simplex formed by two points
        self.simplices = np.array([[0, 1]])

    def find_simplex(self, points: np.ndarray):
        """
        Find simplices containing given points
        """
        points = points.squeeze()
        bound_miss = points > self._bound_max
        bound_miss |= points < self._bound_min

        return np.where(bound_miss, -1, 0)


# ---------------------------------------------------------------------------*/
# - gaussian process sampler

class gaussianprocess_sampler:
    def __init__(self) -> None:
        self._mean = None
        self._kernel = None
        self._discretization = None
        self._noise_var = None
        self._alphas = None

    def _initialize(self, samples_n: int = 1) -> None:
        cov = self._kernel.K(self._discretization) + tf.eye(self._discretization.shape[0], dtype=gpflow.default_float()) * 1e-6

        # sample normal distribution
        samples = np.random.multivariate_normal(np.zeros(self._discretization.shape[0]), cov, size=samples_n)

        cho = linalg.cho_factor(cov, lower=True)
        self._alphas = [
            linalg.cho_solve(cho, samples[[sample_loc], :].T) for sample_loc in range(samples_n)]

    def __call__(self, domain: tf.Tensor) -> tf.Tensor:
        k = self._kernel.K(domain, self._discretization)
        mean = self._mean(domain)
        noise_var = self._noise_var

        def sample(alpha: tf.Tensor) -> tf.Tensor:
            y = mean + tf.matmul(k, alpha)
            if noise_var is not None:
                y += tf.sqrt(noise_var) * tf.random.normal(tf.shape(y), dtype=gpflow.default_float())
            return y

        return [sample(alpha) for alpha in self._alphas]


# ---------------------------------------------------------------------------*/
# - gaussian process

class gaussianprocess:
    def __init__(self, mean_fn, cov_fn: gpflow.kernels.Kernel, dims_i_n: int, obsv_noise_var: float) -> None:

        # create a gaussian process with
        # initial observed zero-data at the origin, i.e. x=0, y=0.
        train_i = tf.zeros((1, dims_i_n), dtype=gpflow.default_float())
        train_o = tf.zeros((1, dims_i_n), dtype=gpflow.default_float())
        self.gp = gpflow.models.GPR(
            (train_i, train_o),
            cov_fn, mean_fn,
            noise_variance=obsv_noise_var)

        self._update_cache()

    def new_sampler(
            self,
            discretization: tf.Tensor,
            samples_n: int = 1, noise_var: float | None = None) -> gaussianprocess_sampler:

        sampler = gaussianprocess_sampler()
        sampler._mean = self.gp.mean_function
        sampler._kernel = self.gp.kernel
        sampler._discretization = discretization
        sampler._noise_var = noise_var
        sampler._initialize(samples_n)

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
        cov = self.gp.kernel.K(train_i) + tf.eye(train_i.shape[0], dtype=gpflow.default_float()) * tf.constant(1e-6, dtype=gpflow.default_float())

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
# - plot triangulation in three-dimensions

def plot3d_triangulation(tri):
    parameters = tri.parameters
    states = tri._domain.states
    simplices = tri.get_simplices(np.arange(tri.nsimplex))

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    ax.plot_trisurf(
        states[:, 0], states[:, 1], parameters[:, 0],
        triangles=simplices)

    return fig, ax
