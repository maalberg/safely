from abc import abstractmethod
from abc import ABCMeta as interface

import numpy as np

from itertools import product as cartesian
from scipy.spatial import Delaunay as delaunay
from scipy.sparse import coo_matrix as sparse_coordinates
import GPy as gpy

import domain as dom
import utilities as utils


# ---------------------------------------------------------------------------*/
# - function

class function(metaclass=interface):
    @abstractmethod
    def __call__(self, domain: np.ndarray, samples_n: int = 1) -> tuple[np.ndarray] | np.ndarray:
        """
        Take ``samples_n`` number of function samples with ``domain`` as input and
        return these samples in a tuple. If ``samples_n`` equals 1, then
        the tuple is dropped and the sample itself is returned.
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


# ---------------------------------------------------------------------------*/
# - uncertainty

class uncertainty(metaclass=interface):
    @abstractmethod
    def evaluate_error(self, domain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the error of this uncertainty in given ``domain`` and return a predicted
        mean value together with a corresponding variance.
        """
        raise NotImplementedError

    @abstractmethod
    def observe_datapoints(self, domain: np.ndarray, value: np.ndarray) -> None:
        """
        Let this uncertainty observe datapoints in ``domain`` with given ``value``.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def datapoints_observed(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Datapoints observed by this uncertainty, see method ``observe_datapoints``.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------*/
# - differentiable function

class differentiable(function):
    @abstractmethod
    def differentiate(self, domain: np.ndarray) -> np.ndarray:
        """
        Differentiate this function on given ``domain`` and return resulting values.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------*/
# - quadratic function

class quadratic(differentiable):
    def __init__(self, parameters: np.ndarray) -> None:
        self._params = np.atleast_2d(parameters)

    def __call__(self, domain: np.ndarray, samples_n: int = 1) -> tuple[np.ndarray] | np.ndarray:
        sample = np.sum(domain.dot(self._params) * domain, axis=1, keepdims=True)
        return sample if samples_n == 1 else [sample for this in range(samples_n)]

    def differentiate(self, domain: np.ndarray) -> np.ndarray:
        return 2 * domain.dot(self._params)

    @property
    def dims_i_n(self) -> int:
        return np.shape(self._params)[1]

    @property
    def dims_o_n(self) -> int:
        return np.shape(self._params)[0]


# ---------------------------------------------------------------------------*/
# - linear function

class linear(function):
    def __init__(self, parameters: list[np.ndarray]) -> None:
        self._parameters = np.column_stack(tuple(map(np.atleast_2d, parameters)))

    def __call__(self, domain: np.ndarray, samples_n: int = 1) -> tuple[np.ndarray] | np.ndarray:
        sample = domain.dot(self._parameters.T)
        return sample if samples_n == 1 else [sample for this in range(samples_n)]

    @property
    def dims_i_n(self) -> int:
        return np.shape(self._parameters)[1]

    @property
    def dims_o_n(self) -> int:
        return np.shape(self._parameters)[0]


# ---------------------------------------------------------------------------*/
# - stochastic function

class stochastic(function, uncertainty):
    pass


# ---------------------------------------------------------------------------*/
# - dynamics

class dynamics(stochastic):
    def __init__(
            self,
            model: function, policy: function = None, error: gpy.kern.Kern = None) -> None:

        self._dims_i_n = model.dims_i_n
        self._dims_o_n = model.dims_o_n

        self.policy = policy

        # if error is present, construct stochastic dynamics,
        # otherwise the dynamics are deterministic
        if error is not None:
            # use model as the mean function of a Gaussian process
            gp_mean = gpy.core.Mapping(model.dims_i_n, model.dims_o_n)
            gp_mean.f = model
            gp_mean.update_gradients = lambda a, b: None

            # a gaussian process with initial observed data at the origin with x=0, y=0
            gp = gpy.core.GP(
                np.zeros((1, model.dims_i_n)), np.zeros((1, model.dims_o_n)),
                error,
                gpy.likelihoods.Gaussian(variance=0), # no observaion noise at the moment
                mean_function=gp_mean)

            # define a sampling method for stochastic dynamics
            def sample(domain: np.ndarray, samples_n: int) -> tuple[np.ndarray] | np.ndarray:
                samples = gp.posterior_samples(domain, size=samples_n)

                # format samples as a tuple of samples
                samples = [samples[..., this_sample] for this_sample in range(samples.shape[-1])]

                # but drop the tuple if there is only one sample requested
                return samples[0] if samples_n == 1 else samples

            self._sampling = sample

            # define a method to evaluate the error of stochastic dynamics
            def error_eval(domain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                return gp.predict(domain)

            self._error = error_eval

            # define a method to observe datapoints in order to reduce the uncertainty of stochastic dynamics
            def observe(domain: np.ndarray, value: np.ndarray) -> None:
                gp.set_XY(
                    np.row_stack([gp.X, domain]),
                    np.row_stack([gp.Y, value]))

            self._observer = observe

            # define a method to return observed datapoints for stochastic dynamics
            def observed() -> tuple[np.ndarray, np.ndarray]:
                return gp.X, gp.Y

            self._observations = observed
        else:
            # define a sampling method for deterministic dynamics
            def sample(domain: np.ndarray, samples_n: int) -> tuple[np.ndarray]:
                sample = model(domain)

                return sample if samples_n == 1 else [sample for this in range(samples_n)]

            self._sampling = sample

            # define a method to evaluate the error of deterministic dynamics (there is no error)
            def error_eval(domain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                mean = model(domain)
                var = np.zeros_like(mean)

                return mean, var

            self._error = error_eval

            # define a method to observe datapoints for deterministic dynamics (this has no meaning)
            def observe(domain: np.ndarray, value: np.ndarray) -> None:
                pass

            self._observer = observe

            # define a method to return observed datapoints in case of deterministic dynamics (there will be none)
            def observed() -> tuple[np.ndarray, np.ndarray]:
                return None

            self._observations = observed

    def __call__(self, domain: np.ndarray, samples_n: int = 1) -> tuple[np.ndarray] | np.ndarray:

        # augment domain with actuation signal if policy is available
        if self.policy is not None: domain = np.column_stack([domain, self.policy(domain)])

        # sample function
        return self._sampling(domain, samples_n)

    def evaluate_error(self, domain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        # augment domain with actuation signal if policy is available
        if self.policy is not None: domain = np.column_stack([domain, self.policy(domain)])

        # evaluate function error
        return self._error(domain)

    def observe_datapoints(self, domain: np.ndarray, value: np.ndarray) -> None:
        # observe given datapoints
        self._observer(domain, value)

    @property
    def dims_i_n(self) -> int:
        return self._dims_i_n

    @property
    def dims_o_n(self) -> int:
        return self._dims_o_n

    @property
    def datapoints_observed(self) -> tuple[np.ndarray, np.ndarray]:
        # return observed datapoints, if any
        return self._observations()


# ---------------------------------------------------------------------------*/
# - inverted pendulum

class pendulum_inv(function):
    def __init__(
            self,
            mass: float, length: float, friction: float,
            policy: function = None,
            normalization: tuple[list, list] = None) -> None:

        self.mass = mass
        self.length = length
        self.friction = friction

        self.policy = policy
        self.normalization = normalization

    def __call__(self, domain: np.ndarray, samples_n: int = 1) -> tuple[np.ndarray] | np.ndarray:

        # augment domain with actuation signal if policy is available
        if self.policy is not None: domain = np.column_stack([domain, self.policy(domain)])

        # execute ordinary differential equation
        state_d = self._execute_ode(domain)

        return state_d if samples_n == 1 else [state_d for this in range(samples_n)]

    def _execute_ode(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:

        state, action = self.denormalize(state, action)

        g = self.gravity
        l = self.length
        f = self.friction
        i = self.inertia

        # physical dynamics
        angle, angular_velocity = np.split(state, indices_or_sections=2, axis=1)
        angular_acceleration = g / l * np.sin(angle) + action / i
        if f > 0: angular_acceleration -= f / i * angular_velocity

        state_d = np.column_stack((angular_velocity, angular_acceleration))
        state_d = self.normalize_state(state_d)

        return state_d

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

        return a, b


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
    def dims_i_n(self) -> int:
        return self._domain.dims_n

    @property
    def dims_o_n(self) -> int:
        return 1

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters
