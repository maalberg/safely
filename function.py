from abc import abstractmethod
from abc import ABCMeta as interface

from scipy import signal
from scipy import linalg

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
    def __call__(self, domain: tf.Tensor) -> tf.Tensor:
        """
        Sample this function on given ``domain``.
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
    def dims_i_n(self) -> int: return self.parameters.shape[1]

    @property
    def dims_o_n(self) -> int: return self.parameters.shape[0]

    def _validate_type(self, domain: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor(domain, dtype=gpflow.default_float()) if isinstance(domain, np.ndarray) else domain


# ---------------------------------------------------------------------------*/
# - differentiable function

class differentiable(function):
    @abstractmethod
    def differentiate(self, domain: tf.Tensor) -> tf.Tensor:
        """
        Differentiate this function on given ``domain``.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------*/
# - linear function

class linear(function):
    def __init__(self, parameters: list[tf.Tensor]) -> None:
        self._parameters = tf.concat(tuple(map(tf.experimental.numpy.atleast_2d, parameters)), axis=1)

    def __call__(self, domain: tf.Tensor) -> tf.Tensor:
        domain = self._validate_type(domain)
        return tf.matmul(domain, self._parameters, transpose_b=True)

    @property
    def parameters(self) -> tf.Tensor: return self._parameters


# ---------------------------------------------------------------------------*/
# - quadratic function

class quadratic(differentiable):
    def __init__(self, parameters: tf.Tensor) -> None:
        self._parameters = tf.experimental.numpy.atleast_2d(parameters)

    def __call__(self, domain: tf.Tensor) -> tf.Tensor:
        domain = self._validate_type(domain)
        return tf.reduce_sum(tf.matmul(domain, self._parameters) * domain, axis=1, keepdims=True)

    def differentiate(self, domain: tf.Tensor) -> tf.Tensor:
        domain = self._validate_type(domain)
        return tf.matmul(domain, self._parameters + tf.transpose(self._parameters))

    @property
    def parameters(self) -> tf.Tensor: return self._parameters


# ---------------------------------------------------------------------------*/
# - stochastic function

class stochastic(function):
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
# - dynamics

class dynamics(stochastic):
    def __init__(
            self,
            model: function, error: gpflow.kernels.Kernel, noise: float,
            policy: function | None = None) -> None:
        """
        Stochastic dynamics are defined by ``model`` and ``error``, where
        the former represents prior knowledge about dynamical behavior, whereas
        the latter describes expected model uncertainty. In order to facilitate the
        sampling of these dynamics, ``domain_sampling`` defines the discretization of
        the sampling. Finally, ``policy`` can be set at a later stage, which allows swapping policies.
        """

        # save parameters of given mean model to return as parameters of this class
        self._parameters = model.parameters

        # make policy a publicly available property of this class
        self.policy = policy

        # these stochastic dynamics are internally implemented in terms of a gaussian process
        self._gp = utils.gaussianprocess(
            model, error,
            dims_n=(model.dims_i_n, model.dims_o_n),
            obsv_noise_var=noise)

        self._gp_sampler = None

    def initialize_sampler(
            self,
            discretization: tf.Tensor,
            samples_n: int = 1, noise_var: float = 0.001**2) -> None:
        """
        Initialize a new dynamics sampler given sampling ``discretization``, the number of
        samples requested ``samples_n`` and sampling, or observation,
        noise variance ``noise_var``.
        """
        self._gp_sampler = self._gp.new_sampler(discretization, samples_n, noise_var)

    def __call__(self, domain: tf.Tensor, with_noise: bool = False) -> tf.Tensor:
        """
        Sample this function on given ``domain`` with the possibility to add observation
        noise when ``with_noise`` parameter is true.
        """
        domain = self._validate_type(domain)

        # augment domain with actuation signal if policy is available
        if self.policy is not None: domain = tf.concat([domain, self.policy(domain)], axis=1)

        # sample function
        return self._gp_sampler(domain, with_noise)

    def evaluate_error(self, domain: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        domain = self._validate_type(domain)

        # augment domain with actuation signal if policy is available
        if self.policy is not None: domain = tf.concat([domain, self.policy(domain)], axis=1)

        # evaluate function error
        return self._gp.predict(domain)

    def observe_datapoints(self, domain: tf.Tensor, value: tf.Tensor) -> None:
        domain = self._validate_type(domain)
        value = self._validate_type(value)

        # observe given datapoints
        self._gp.update_data(domain, value)

    @property
    def datapoints_observed(self) -> tuple[tf.Tensor, tf.Tensor]:
        return self._gp.train_i, self._gp.train_o

    @property
    def parameters(self) -> tf.Tensor: return self._parameters


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

    def __call__(self, domain: tf.Tensor) -> tf.Tensor:

        # make sure there is at least one row,
        # i.e. state (possibly with action), in domain argument
        domain = np.atleast_2d(domain)

        # augment domain with actuation signal if policy is available
        if self.policy is not None: domain = tf.concat([domain, self.policy(domain)], axis=1)

        # call internal dynamics
        return self._solve_ode(domain)

    def _solve_ode(self, domain: tf.Tensor) -> tf.Tensor:
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

    def __call__(self, domain: np.ndarray) -> np.ndarray:
        value = self._func(domain)
        return tf.clip_by_value(value, -self._clip, self._clip)

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

    def __call__(self, domain: tf.Tensor) -> tf.Tensor:

        # evaluate the means of all stochastic functions inside the list
        return tf.concat([func.evaluate_error(domain)[0] for func in self._funcs], axis=1)

    @property
    def parameters(self) -> tf.Tensor:
        return tf.constant([func.parameters for func in self._funcs])

    @property
    def dims_i_n(self) -> int:
        return self._funcs[0].dims_i_n

    @property
    def dims_o_n(self) -> int:
        return tf.reduce_sum([func.dims_o_n for func in self._funcs])


# ---------------------------------------------------------------------------*/
# - triangulation as a function approximator

class triangulation(function):
    def __init__(self, domain: dom.gridworld, values: tf.Tensor) -> None:
        """
        This triangulation will approximate a function on the given ``domain`` and
        is parameterized with ``values``, where every row holds a value
        corresponding to a data point from the ``domain``.
        """

        # since the minimum number of dimensions
        # supported by SciPy implementation of Delaunay algorithm is 2, then check given domain
        if len(domain.step) == 1:
            # define two points of a 1D unit hyper-rectangle
            # and instantiate a corresponding delaynay triangulation
            unit_points = np.array([[0], domain.step])
            tri = utils.delaunay_1d(unit_points)
        else:
            # based on domain discretization in every dimension,
            # define the limits of a unit hyper-rectangle
            unit_lim = np.diag(domain.step)

            # use a cartesian product to derive the corresponding vertices (points)
            unit_points = tf.convert_to_tensor(
                list(cartesian(*unit_lim)),
                dtype=gpflow.default_float())

            # perform triangulation of the unit hyper-rectangle
            tri = utils.delaunay_nd(unit_points)

        simplices_map = self._create_simplices_map(domain, tri)
        hyperplanes_mat = self._create_hyperplanes_mat(domain, simplices_map)

        self.nsimplex = tri.nsimplex * domain.rectangles_n

        # make sure there is at least one row in given values
        parameters = tf.experimental.numpy.atleast_2d(values)

        self._hyperplanes_mat = hyperplanes_mat
        self._simplices_map = simplices_map
        self._parameters = parameters
        self._domain = domain
        self._tri = tri

    def _create_simplices_map(self, domain: dom.gridworld, triangulation: utils.delaunay) -> tf.Tensor:
        """
        Map simplices from internal unit ``triangulation`` to original ``domain`` and
        return the indices of mapped simplices.
        """

        # get simplices from the internal triangulation of a unit domain
        #
        # the simplices contain the indices of points which form
        # these simplices
        simplices_unit = triangulation.simplices

        # extract points that form unit domain simplices
        #
        # The points are reshaped to remove simplex-based partitioning of points,
        # e.g. in a two-dimensional case a unit hyper-rectangle will
        # have two simplices with 3 two-dimensional points each.
        # So the idea is to change shape (2, 3, 2) into shape
        # (6, 2), such there are 6 two-domensional states
        # that can be further analyzed.
        points_unit = tf.reshape(
            tf.gather(triangulation.points, indices=simplices_unit),
            [-1, domain.dims_n])

        # locate states [their indices] in the original domain,
        # which correspond to the points inside the internal unit hyper-rectangle
        #
        # The original state indices are then reshaped to follow the unit domain
        # simplex structure.
        return tf.reshape(
            domain.locate_points(points_unit + domain.offset),
            shape=simplices_unit.shape)

    def find_simplex(self, points: tf.Tensor) -> tf.Tensor:
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
        points_unit = self._domain.shift_points(points, needs_clipping=True) % self._domain.step

        # find which points belong to which triangle inside a unit hyper-rectangle
        simplices_unit = tf.experimental.numpy.atleast_1d(self._tri.find_simplex(points_unit))

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
        return simplices_unit + rectangles * self._tri.nsimplex

    def simplices(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Given ``indices`` of simplices, get the indices of points that form these
        simplices in the original domain. Each row of the returned
        array contains the indices of simplex corners.
        """

        # convert the given original domain indices/locations to the unit domain ones
        indices_unit = tf.math.floormod(indices, self._tri.nsimplex)

        # extract unit simplices, which contain indices pointing to the original domain
        simplices = tf.gather(self._simplices_map, indices=indices_unit)

        # locate the upper-left corners of rectangles in the original domain that
        # correspond to given simplex indices
        #
        # convertion of indices to rectangles is based on the fact that
        # a rectangle will contain nsimplex simplices, so
        # we can perform the corresponding division
        rectangles_origins = self._domain.locate_origins(tf.math.floordiv(indices, self._tri.nsimplex))
        if simplices.ndim > 1:
            # add extra inner-most dimension
            rectangles_origins = tf.expand_dims(rectangles_origins, axis=-1)

        return tf.add(simplices, rectangles_origins)

    def _create_hyperplanes_mat(self, domain: dom.gridworld, simplices: tf.Tensor) -> tf.Tensor:
        """
        Create hyperplane matrices. There are as many hyperplanes as there are ``simplices``,
        and each hyperplane is represented by a n-by-n matrix of coefficients,
        where n is equal to the number of ``domain`` dimensions.
        """

        # first, an empty list is created to be later filled with hyperplane matrices
        hyperplanes_mat = []

        # compute equation coefficients for every hyperplane (simplex)
        for simplex in simplices:

            # get states, or points, in the original domain that form the current simplex,
            # i.e. lie on this hyperplane
            simplex_points = domain.get_points(simplex)

            # subtract subsequent points from the first one to get vectors which lie in the plane, e.g.
            # for two dimensions there will be three points of a triangular plane
            # and, thus, two resulting vectors lying in this plane
            simplex_vectors = simplex_points[1:] - simplex_points[:1]

            # the simplex vectors above form a linear system of equations, e.g.
            # there will be two equations of the form ax1 + bx2 = c for a two-dimensional
            # case, so you get a 2-by-2 matrix A of coefficients. This matrix can be inverted to solve
            # a linear program, such as A^-1 A x = A^-1 b => x = A^-1 b
            hyperplanes_mat.append(tf.linalg.inv(simplex_vectors))

        # pack the filled list into an output tensor
        return tf.stack(hyperplanes_mat)

    def _get_hyperplanes(self, points: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Get equations of hyperplanes that contain given ``points``. In general, these
        equations have a linear form Ax = b, where A is a set of hyperplane
        coefficients, and where b is an offset of a point from its origin
        inside the hyperplane, computed as points - origins. Then, x
        are barycentric coordinates of the given points inside the
        hyperplane. Following this, the method returns an
        inverted matrix A^{-1} and the origins of points
        to compute values of x as x = A^{-1} * (points - origins).
        In addition, the method returns simplices that contain given ``points``.
        """

        # find simplices in the original domain [their indices actually]
        # that contain given points
        simplices_loc = self.find_simplex(points)

        # extract simplices from the original domain at the found indices,
        # such that every simplex [basically a triangle]
        # is a row with three point indices
        simplices = self.simplices(simplices_loc)

        # having the indices of points that form simplices, it is
        # possible to retrieve those points that are
        # considered origins inside the simplices
        origins = self._domain.get_points(simplices[:, 0])

        # finally, to get hyperplane matrices convert [note the modulus operation] the indices
        # of simplices in the original domain to the ones in the unit domain..
        unit_simplices = simplices_loc % self._tri.nsimplex

        # ..and get hyperplane matrices
        hyperplanes = tf.gather(self._hyperplanes_mat, indices=unit_simplices)

        return hyperplanes, origins, simplices

    def __call__(self, domain: tf.Tensor) -> tf.Tensor:

        # make sure domain has at least one row, i.e. one data point to sample
        points = tf.experimental.numpy.atleast_2d(domain)

        # clip any out-of-bound points to discretization limits
        points = tf.clip_by_value(
            points,
            self._domain.dims_lim[:, 0], self._domain.dims_lim[:, 1])

        # get hyperplane geometry required to compute x = A^{-1} * (points - origins),
        # where x are barycentric coordinates of the given points
        hyperplanes, origins, simplices = self._get_hyperplanes(points)

        # compute offsets of points from their origins
        offsets = points - origins

        # compute weights as barycentric coordinates of points inside this triangulation
        #
        # The dot product computation of A^{-1} * b first produces a result with size
        # (number of points, number of domain dimensions), e.g. for a two-
        # dimensional case the size will be (1, 2) for a single point.
        # But the triangulation of a two-dimensional space is built
        # from triangles, so there must be one more dimension to
        # accomodate the last triangle vertex. And since the
        # coordinates, or weights, are normalized, then
        # the last coordinate can be computed as 1 - sum of others.
        coords_others = tf.reduce_sum(offsets[:, :, tf.newaxis] * hyperplanes, axis=1)
        coords_first = 1 - tf.reduce_sum(coords_others, axis=1, keepdims=True)
        weights = tf.concat((coords_first, coords_others), axis=1)

        # based on determined simplices, gather parameters,
        # i.e. function values at the vertices of this triangulation
        params = tf.gather(self.parameters, indices=simplices)

        # having weights, normalized from 0 to 1, and function values, or parameters, at the vertices
        # of this triangulation, it is straightforward to scale the values by the weights,
        # thus approximating function at the given arbitrary locations, or points
        return tf.reduce_sum(weights[:, :, tf.newaxis] * params, axis=1)

    @property
    def parameters(self) -> np.ndarray: return self._parameters

    @property
    def dims_i_n(self) -> int: return self._domain.dims_n

    @property
    def dims_o_n(self) -> int: return 1


# ---------------------------------------------------------------------------*/
# - neural network

class neuralnetwork(function):
    def __init__(self, dims_i_n: int, layers: list[int], activations: list[str]) -> None:

        # when working with gpflow, the data type of tensorflow must match
        # the default one of gpflow
        dtype = gpflow.default_float()

        # construct a sequential model to wrap network layers
        self._model = tf.keras.Sequential()

        # specify input data shape;
        # this also makes the model build its layers, and thus weights, automatically
        self._model.add(tf.keras.Input(shape=(dims_i_n,), dtype=dtype))

        kernel_init = 'glorot_uniform'

        # add hidden layers
        for layer, activation in zip(layers[:-1], activations[:-1]):
            self._model.add(
                tf.keras.layers.Dense(
                    layer,
                    activation=activation,
                    kernel_initializer=kernel_init,
                    use_bias=True,
                    dtype=dtype))

        # add output layer
        self._model.add(
            tf.keras.layers.Dense(
                layers[-1],
                activation=activations[-1],
                kernel_initializer=kernel_init,
                use_bias=False,
                dtype=dtype))

    def __call__(self, domain: tf.Tensor) -> tf.Tensor:
        domain = self._validate_type(domain)

        return self._model(domain)

    @property
    def parameters(self) -> tf.Tensor:
        return self._model.trainable_variables
