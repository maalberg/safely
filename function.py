from abc import abstractmethod
from abc import ABCMeta as interface

from scipy import signal
from scipy.spatial import Delaunay as delaunay

from itertools import product as cartesian

import numpy as np
import tensorflow as tf
import gpflow

import domain as dom
import utilities as utils


# ---------------------------------------------------------------------------*/
# - function

class function(metaclass=interface):
    @abstractmethod
    def __call__(self, domain: tf.Tensor, samples_n: int = 1) -> list[tf.Tensor] | tf.Tensor:
        """
        Take ``samples_n`` number of function samples with ``domain`` as input and
        return these samples in a list. If ``samples_n`` equals 1, then
        the list is dropped and the sample itself is returned.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> tf.Tensor:
        """
        Parameters of this function.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dims_i_n(self) -> int:
        """
        Number of input dimensions.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dims_o_n(self) -> int:
        """
        Number of output dimensions.
        """
        raise NotImplementedError

    def _validate_type(self, domain: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor(domain, dtype=gpflow.default_float()) if isinstance(domain, np.ndarray) else domain


# ---------------------------------------------------------------------------*/
# - uncertainty

class uncertainty(metaclass=interface):
    @abstractmethod
    def evaluate_error(self, domain: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Evaluate the error of this uncertainty in given ``domain`` and return a predicted
        mean value together with a corresponding variance.
        """
        raise NotImplementedError

    @abstractmethod
    def observe_datapoints(self, domain: tf.Tensor, value: tf.Tensor) -> None:
        """
        Let this uncertainty observe datapoints in ``domain`` with given ``value``.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def datapoints_observed(self) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Datapoints observed by this uncertainty, see method ``observe_datapoints``.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------*/
# - differentiable function

class differentiable(function):
    @abstractmethod
    def differentiate(self, domain: tf.Tensor) -> tf.Tensor:
        """
        Differentiate this function on given ``domain`` and return resulting values.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------*/
# - quadratic function

class quadratic(differentiable):
    def __init__(self, parameters: tf.Tensor) -> None:
        self._parameters = tf.experimental.numpy.atleast_2d(parameters)

    def __call__(self, domain: tf.Tensor, samples_n: int = 1) -> list[tf.Tensor] | tf.Tensor:
        domain = self._validate_type(domain)
        return tf.reduce_sum(tf.matmul(domain, self._parameters) * domain, axis=1, keepdims=True)

    def differentiate(self, domain: tf.Tensor) -> tf.Tensor:
        domain = self._validate_type(domain)
        return tf.matmul(domain, self._parameters + tf.transpose(self._parameters))

    @property
    def parameters(self) -> tf.Tensor:
        return self._parameters

    @property
    def dims_i_n(self) -> int:
        return self._parameters.shape[1]

    @property
    def dims_o_n(self) -> int:
        return self._parameters.shape[0]


# ---------------------------------------------------------------------------*/
# - linear function

class linear(function):
    def __init__(self, parameters: list[tf.Tensor]) -> None:
        self._parameters = tf.concat(tuple(map(tf.experimental.numpy.atleast_2d, parameters)), axis=1)

    def __call__(self, domain: tf.Tensor, samples_n: int = 1) -> list[tf.Tensor] | tf.Tensor:
        domain = self._validate_type(domain)

        sample = tf.matmul(domain, self._parameters, transpose_b=True)

        return sample if samples_n == 1 else [sample for this in range(samples_n)]

    @property
    def parameters(self) -> tf.Tensor:
        return self._parameters

    @property
    def dims_i_n(self) -> int:
        return self._parameters.shape[1]

    @property
    def dims_o_n(self) -> int:
        return self._parameters.shape[0]


# ---------------------------------------------------------------------------*/
# - stochastic function

class stochastic(function, uncertainty):
    pass


# ---------------------------------------------------------------------------*/
# - dynamics

class dynamics(stochastic):
    class gpr:
        def __init__(
                self,
                mean_fn: function, cov_fn: gpflow.kernels.Kernel) -> None:

            # create a gaussian process with
            # initial observed data at the origin with x=0, y=0.
            train_i = tf.zeros((1, mean_fn.dims_i_n), dtype=gpflow.default_float())
            train_o = tf.zeros((1, mean_fn.dims_i_n), dtype=gpflow.default_float())
            self.gp = gpflow.models.GPR((train_i, train_o), cov_fn, mean_fn)

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
            self.cholesky = tf.linalg.cholesky(cov)

            # ... and compute the product between the cholesky and the outputs, i.e.
            # alpha = (K + sigma^2 * I)^-1 * y, see (2.25) from Rasmussen & Williams, 2006
            self.alpha = tf.linalg.triangular_solve(self.cholesky, train_o, lower=True)

        def predict(self, test_i: tf.Tensor, full_cov: bool = False) -> tuple[tf.Tensor, tf.Tensor]:
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
            mean = self.gp.mean_function(test_i)
            cov = self.gp.kernel.K(self.train_i, test_i)

            a = tf.linalg.triangular_solve(self.cholesky, cov, lower=True)
            mean = tf.matmul(a, self.alpha, transpose_a=True) + mean

            if full_cov:
                cov = self.gp.kernel.K(test_i)
                var = cov - tf.matmul(a, a, transpose_a=True)
                shape = tf.stack([1, 1, tf.shape(self.train_o)[1]])
                var = tf.tile(tf.expand_dims(var, 2), shape)
            else:
                cov = self.gp.kernel.K_diag(test_i)
                var = cov - tf.reduce_sum(tf.square(a), axis=0)
                var = tf.tile(tf.reshape(var, (-1, 1)), [1, tf.shape(self.train_o)[1]])

            return mean, var

        def update_data(self, train_i: tf.Tensor, train_o: tf.Tensor) -> None:
            # combine existing and new training data together
            train_i = tf.concat([self.train_i, train_i], axis=0)
            train_o = tf.concat([self.train_o, train_o], axis=0)

            self.gp.data = train_i, train_o
            self._update_cache()

        @property
        def train_i(self) -> tf.Tensor:
            return self.gp.data[0]

        @property
        def train_o(self) -> tf.Tensor:
            return self.gp.data[1]

    def __init__(
            self,
            model: function, error: gpflow.kernels.Kernel,
            policy: function | None = None) -> None:
        """
        Optional parameter ``policy`` can be set at a later stage,
        which allows swapping policies to conduct various experiments.
        """

        self._dims_i_n = model.dims_i_n
        self._dims_o_n = model.dims_o_n

        # save parameters of given mean model to return as parameters of this class
        self._parameters = model.parameters

        # make policy a publicly available property of this class
        self.policy = policy

        self.gp = dynamics.gpr(model, error)

    def __call__(self, domain: tf.Tensor, samples_n: int = 1) -> list[tf.Tensor] | tf.Tensor:
        domain = self._validate_type(domain)

        # augment domain with actuation signal if policy is available
        if self.policy is not None: domain = tf.stack([domain, self.policy(domain)], axis=1)

        # sample function
        return self._sample_gp(domain, samples_n)
    
    def _sample_gp(self, domain: tf.Tensor, samples_n: int) -> tuple[tf.Tensor] | tf.Tensor:
        mean, cov = self.gp.predict(domain, full_cov=True)

        samples = np.random.multivariate_normal(
            tf.squeeze(mean, axis=-1), tf.squeeze(cov, axis=-1), size=samples_n)

        samples = tf.expand_dims(samples, axis=-1)

        # format samples as a list of samples [the first dimension contains the number of samples]
        samples = [samples[this_sample, ...] for this_sample in range(samples.shape[0])]

        # but drop the list if there is only one sample requested
        return samples[0] if samples_n == 1 else samples

    def evaluate_error(self, domain: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        domain = self._validate_type(domain)

        # augment domain with actuation signal if policy is available
        if self.policy is not None: domain = tf.stack([domain, self.policy(domain)], axis=1)

        # evaluate function error
        return self.gp.predict(domain)

    def observe_datapoints(self, domain: tf.Tensor, value: tf.Tensor) -> None:
        domain = self._validate_type(domain)
        value = self._validate_type(value)

        # observe given datapoints
        self.gp.update_data(domain, value)

    @property
    def datapoints_observed(self) -> tuple[tf.Tensor, tf.Tensor]:
        return self.gp.train_i, self.gp.train_o

    @property
    def parameters(self) -> tf.Tensor: return self._parameters

    @property
    def dims_i_n(self) -> int: return self._dims_i_n

    @property
    def dims_o_n(self) -> int: return self._dims_o_n


# ---------------------------------------------------------------------------*/
# - inverted pendulum

class pendulum_inv(function):
    def __init__(
            self,
            mass: float, length: float, friction: float = 0.0,
            normalization: tuple[list, list] = None,
            timestep: float = 0.01) -> None:

        self.mass = mass
        self.length = length
        self.friction = friction

        self.timestep = timestep
        self.normalization = normalization

        # policy is initially set to none, because the user may
        # first require to instantiate this class in order to obtain a
        # linearized model for control, and only after that the user can assign a proper policy
        self.policy = None

    def __call__(self, domain: np.ndarray, samples_n: int = 1) -> tuple[np.ndarray] | np.ndarray:

        # make sure there is at least one row,
        # i.e. state (possibly with action), in domain argument
        domain = np.atleast_2d(domain)

        # augment domain with actuation signal if policy is available
        if self.policy is not None: domain = np.column_stack([domain, self.policy(domain)])

        # call internal dynamics
        value = self._solve_ode(domain)

        # repeat values (samples) to respect the protocol of this function
        return value if samples_n == 1 else [value for this in range(samples_n)]

    def _solve_ode(self, domain: np.ndarray) -> np.ndarray:
        """
        Solve the ordinary differential equation of this pendulum given ``domain`` as input.
        It is expected that ``domain`` contains the denormalized state of
        the pendulum, as well as the denormalized control input.
        This method uses Euler's integration to
        compute the next state.
        """

        # extract state and actions from given domain and denormalize them
        state, action = np.split(domain, indices_or_sections=[2], axis=1)
        state, action = self.denormalize(state, action)

        # use Euler's method to solve the ordinary differential equation of this pendulum
        integration_steps_n = 10
        dt = self.timestep / integration_steps_n
        for step in range(integration_steps_n):
            state_derivative = self._differentiate(state, action)
            state = state + dt * state_derivative

        # normalize the state back and return it
        state = self.normalize_state(state)
        return state

    def _differentiate(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Based on denormalized ``state`` and ``action``, use the ordinary differential equation
        of this inverted pendulum to compute a time derivative of
        the pendulum's dynamics.
        """
        g = self.gravity
        l = self.length
        f = self.friction
        i = self.inertia

        # calculate the time-derivative of the current state
        # using the ordinary differential equation of this pendulum
        angle, angular_velocity = np.split(state, indices_or_sections=2, axis=1)
        angular_acceleration = g / l * np.sin(angle) + action / i
        if f > 0: angular_acceleration -= f / i * angular_velocity

        return np.column_stack((angular_velocity, angular_acceleration))

    @property
    def parameters(self) -> np.ndarray:
        return np.array([self.mass, self.length, self.friction])

    @property
    def dims_i_n(self) -> int:
        return 3

    @property
    def dims_o_n(self) -> int:
        return 2

    @property
    def inertia(self):
        return self.mass * self.length ** 2

    @property
    def gravity(self):
        return 9.81

    def _get_state_denorm(self) -> np.ndarray:
        return np.diag(np.atleast_1d(self.normalization[0]))

    def _get_state_norm(self) -> np.ndarray:
        return np.diag(np.diag(self._get_state_denorm()) ** -1)

    def _get_action_denorm(self) -> np.ndarray:
        return np.diag(np.atleast_1d(self.normalization[1]))

    def _get_action_norm(self) -> np.ndarray:
        return np.diag(np.diag(self._get_action_denorm()) ** -1)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        if self.normalization is None:
            return state

        return state.dot(self._get_state_norm())

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        if self.normalization is None:
            return action

        return action.dot(self._get_action_norm())

    def normalize(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.normalize_state(state), self.normalize_action(action)

    def denormalize_state(self, state: np.ndarray) -> np.ndarray:
        if self.normalization is None:
            return state

        return state.dot(self._get_state_denorm())

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        if self.normalization is None:
            return action

        return action.dot(self._get_action_denorm())

    def denormalize(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.denormalize_state(state), self.denormalize_action(action)

    def linearize(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Linearize this pendulum and return discretized system and input matrices.
        Provided normalization has been specified, the
        returned matrices are normalized as well.
        """
        g = self.gravity
        l = self.length
        f = self.friction
        i = self.inertia

        # linearized dynamics, where sinx = x
        a = np.array([
            [0, 1],
            [g / l, -f / i]])

        # action input
        b = np.array([
            [0],
            [1 / i]])

        # provided the maximum values of states and actions are available,
        # normalize linearized matrices, adhering to the following signal scheme
        #
        # normalized output <- normalize * matrix * denormalize <- normalized input
        if self.normalization is not None:
            state_norm = self._get_state_norm()

            a = np.linalg.multi_dot((state_norm, a, self._get_state_denorm()))
            b = np.linalg.multi_dot((state_norm, b, self._get_action_denorm()))

        # discretize this pendulum based on time step
        c = np.eye(2) # output matrix propagates both system states to the output
        d = np.zeros((2, 1)) # there is no direct feedthrough of control input to system output
        model_continuous = signal.StateSpace(a, b, c, d)
        model_discrete = model_continuous.to_discrete(self.timestep)

        return model_discrete.A, model_discrete.B


# ---------------------------------------------------------------------------*/
# - decorator to saturate the output of a function

class saturated(function):
    def __init__(self, func: function, clip: float) -> None:
        self._func = func
        self._clip = clip

    def __call__(self, domain: np.ndarray, samples_n: int = 1) -> tuple[np.ndarray] | np.ndarray:
        value = self._func(domain, samples_n)
        return np.clip(value, -self._clip, self._clip)

    @property
    def parameters(self) -> np.ndarray:
        self._func.parameters

    @property
    def dims_i_n(self) -> int:
        return self._func.dims_i_n

    @property
    def dims_o_n(self) -> int:
        return self._func.dims_o_n


# ---------------------------------------------------------------------------*/
# - one-dimensional stochastic functions stacked to form a vector function

class stochastic_stacked(function):
    def __init__(self, functions: list[stochastic]) -> None:
        self._funcs = functions

    def __call__(self, domain: np.ndarray, samples_n: int = 1) -> tuple[np.ndarray] | np.ndarray:

        # evaluate the means of all stochastic functions inside the list
        sample = np.column_stack([func.evaluate_error(domain)[0] for func in self._funcs])

        return sample if samples_n == 1 else [sample for this in range(samples_n)]

    @property
    def parameters(self) -> np.ndarray:
        return np.ndarray([func.parameters for func in self._funcs])

    @property
    def dims_i_n(self) -> int:
        return self._funcs[0].dims_i_n

    @property
    def dims_o_n(self) -> int:
        return np.sum([func.dims_o_n for func in self._funcs])


# ---------------------------------------------------------------------------*/
# - triangulation as a function approximator

class triangulation(function):
    def __init__(self, domain: dom.gridworld, values: np.ndarray) -> None:
        """
        This triangulation will approximate a function on the given ``domain`` and
        is parameterized with ``values``, where every row holds a value
        corresponding to a data point from the ``domain``
        """
        self._domain = domain

        # make sure there is at least one row in given values
        self._parameters = np.atleast_2d(values)

        # since the minimum number of dimensions
        # supported by SciPy implementation of Delaunay algorithm is 2, check given domain
        if len(self._domain.step) == 1:
            # define two points of a 1D unit hyper-rectangle
            # and instantiate a corresponding delaynay triangulation
            unit_points = np.array([[0], self._domain.step])
            self._tri = utils.delaunay1d(unit_points)
        else:
            # based on domain discretization in every dimension,
            # define the limits of a unit hyper-rectangle
            unit_lim = np.diag(self._domain.step)

            # use a cartesian product to derive the corresponding vertices (points)
            unit_points = np.array(list(cartesian(*unit_lim)))

            # perform triangulation of the unit hyper-rectangle
            self._tri = delaunay(unit_points)

        self._unit_simplices = self._map_simplices(self._tri, self._domain)
        self._hyperplanes = self._init_hyperplanes(
            self._tri,
            self._unit_simplices,
            self._domain)

        self.nsimplex = self._tri.nsimplex * self._domain.rectangles_n

    def _map_simplices(self, triangulation, domain: dom.gridworld) -> np.ndarray:
        """
        Derive simplices from the ``triangulation`` of a unit hyper-rectangle, which
        contains point indices mapped to the original ``domain``.

        The function returns the remapped simplices.
        """

        # get simplices from the internal triangulation of a unit domain
        #
        # the simplices contain the indeces of points which form
        # these simplices
        unit_simplices = triangulation.simplices

        # locate states [their indices] in the original domain,
        # which correspond to the points inside the internal unit hyper-rectangle
        #
        # note that such interdomain localisation is possible, due to
        # the regulatory of the original domain grid
        original_indices = domain.locate_states(
            triangulation.points + domain.offset)

        unit_simplices_mapped = np.empty_like(unit_simplices)

        # replace the indices inside the unit simplices with original ones
        for this, index in enumerate(original_indices):
            unit_simplices_mapped[unit_simplices == this] = index

        return unit_simplices_mapped

    def _init_hyperplanes(self, triangulation, simplices: np.ndarray, domain: dom.gridworld) -> np.ndarray:
        """
        Initialize hyperplane equations based on ``triangulation`` of a unit hyper-rectangle,
        ``simplices`` mapped into the original domain and
        the original ``domain`` itself.

        If we consider a linear program Ax = b, then this method returns A^-1, which can then be
        used to find optimal x = A^-1 b .
        """
        # there are as many hyperplanes as there are simplices in our unit hyper-rectangle,
        # then each hyperplane is represented by a n-by-n matrix of coefficients,
        # where n is equal to the number of input dimensions
        hyperplanes = np.empty((triangulation.nsimplex, domain.dims_n, domain.dims_n))

        # compute equation coefficients for every hyperplane (simplex)
        for this, simplex in enumerate(simplices):

            # get states, or points, in the original domain that form the current simplex,
            # i.e. lie on this hyperplane
            simplex_points = domain.get_states(simplex)

            # subtract subsequent points from the first one to get vectors which lie in the plane, e.g.
            # for two dimensions there will be three points of a triangular plane
            # and, thus, two resulting vectors lying in this plane
            simplex_vectors = simplex_points[1:] - simplex_points[:1]

            # the simplex vectors above form a linear system of equations, e.g.
            # there will be two equations of the form ax1 + bx2 = c for a two-dimensional
            # case, so you get a 2-by-2 matrix A of coefficients. This matrix can be inverted to solve
            # a linear program, such as A^-1 A x = A^-1 b => x = A^-1 b
            hyperplanes[this] = np.linalg.inv(simplex_vectors)

        return hyperplanes

    def find_simplex(self, points: np.ndarray) -> np.ndarray:
        """
        Find simplices, or triangles, which contain given ``points`` and
        return the indices of these simplices w.r.t. to the original domain.
        """

        # convert given points to the domain of a unit hyper-rectangle
        #
        # > the points are first shifted from the original domain to a
        #   zero-origin domain, i.e. [0, point_max - original_domain_offset], and then
        #
        # > the points are scaled down (note the modulus operation) to the unit hyper-rectangle,
        #   whose size is denoted by the discretization of the original domain
        unit_points = self._domain.shift_states(points, needs_clipping=True) % self._domain.step

        # find which points belong to which triangle inside a unit hyper-rectangle
        unit_simplices = np.atleast_1d(self._tri.find_simplex(unit_points))

        # locate rectangles in the original domain, which are closest to given points
        rectangles = self._domain.locate_rectangles(points)

        # propagate unit simplices to rectangles in the original domain, e.g.
        # in case of 20-by-20 points 2D grid there will be 19-by-19,
        # or 361, rectangles; at the same time a unit hyper-
        # rectangle will consist of 2 simplices,
        # so we should get 361*2=722 simplices
        # in the original domain, note
        # the multiplication by
        # nsimplex below.
        return unit_simplices + rectangles * self._tri.nsimplex

    def get_simplices(self, indices: np.ndarray) -> np.ndarray:
        """
        Get the indices of points forming every simplex in the original domain
        that resides at given ``indices``.
        """

        # convert given original domain indices to the unit domain ones
        unit_indices = np.remainder(indices, self._tri.nsimplex)

        # extract unit simplices, which contain indices pointing to the original domain
        simplices = self._unit_simplices[unit_indices].copy()

        # locate the upper-left corners of rectangles in the original domain that
        # correspond to given simplex indices
        #
        # convertion of indices to rectangles is based on the fact that
        # a rectangle will contain nsimplex simplices, so
        # we can perform the corresponding division
        rectangles_origins = self._domain.locate_origins(
            np.floor_divide(indices, self._tri.nsimplex))

        if simplices.ndim > 1:
            rectangles_origins = rectangles_origins[:, np.newaxis] # add extra dimension

        simplices += rectangles_origins
        return simplices

    def _get_weights(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate a linear combination of triangulation weights for given ``points`` and
        return these weights together with simplices, in
        which the given ``points`` reside.
        """

        # find simplices in the original domain [their indices actually],
        # in which given points reside
        points_simplices = self.find_simplex(points)

        # extract simplices from the original domain at the found indices,
        # such that every simplex [basically a triangle]
        # is a row with three indices
        simplices = self.get_simplices(points_simplices)

        # convert [note the modulus operation] the indeces of simpleces in the original domain
        # to the ones in the unit domain
        unit_simpleces = points_simplices % self._tri.nsimplex

        # geometrically a hyperplane ax = b can be interpreted as a set of points x with
        # a constant inner product with a given vector a, and b is then the
        # inner product constant showing an offset from the origin.
        #
        # our intention here is to compute optimal weights x, based on the linear equation
        # of a hyperplane, i.e. x = A^-1 * b
        #
        # therefore, first of all get hyperplane coefficients, or A^-1,
        # corresponding to given unit simplices
        hyperplanes = self._hyperplanes[unit_simpleces]

        # next, in order to have the constant b, we need to know the origins
        origins = self._domain.get_states(simplices[:, 0])

        # prepare hyperplane points by clipping them
        points = np.clip(
            points,
            self._domain.dims_lim[:, 0],
            self._domain.dims_lim[:, 1])

        # .. and determine the offset of points from the origin
        offset = points - origins

        # prepare an empty array to store weights
        #
        # array size is specified as (number of points, number of domain dimensions + 1), e.g.
        # in a two-dimensional case a simplex is a triangle with three points, so
        # number of domain dimensions + 1 yields 3.
        weights = np.empty((len(points), self._domain.dims_n + 1))

        # take a dot product x = A^-1 * b to compute weights
        #
        # in a two-dimensional case, the dot product produces two dimensions as well, yet
        # we have an array with three dimensions, so we write the result
        # the last two dimensions of the weights array.
        np.sum(offset[:, :, np.newaxis] * hyperplanes, axis=1, out=weights[:, 1:])

        # now we still need to fill the first dimension of the weights array,
        # so we use the property of affine sets where the coefficients
        # of points sum to one, see Boyd and Vandenberghe, 2004.
        weights[:, 0] = 1 - np.sum(weights[:, 1:], axis=1)

        return weights, simplices

    def __call__(self, domain: np.ndarray, samples_n: int = 1) -> tuple[np.ndarray] | np.ndarray:

        # TODO: document!

        # make sure state has at least one row, i.e. one data point to sample
        domain = np.atleast_2d(domain)

        weights, simplices = self._get_weights(domain)

        # sample function values
        parameters = self._parameters[simplices]

        return np.sum(weights[:, :, np.newaxis] * parameters, axis=1).reshape(-1, 1)

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    @property
    def dims_i_n(self) -> int:
        return self._domain.dims_n

    @property
    def dims_o_n(self) -> int:
        return 1
