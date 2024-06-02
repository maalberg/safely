from abc import abstractmethod
from abc import ABCMeta as interface

import numpy as np
import scipy as sp

from itertools import product as cartesian
from scipy.spatial import Delaunay as delaunay
from scipy.sparse import coo_matrix as sparse_coordinates
import GPy as gpy

import domain as dom
import utilities as util


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
# differentiable function

class differentiable(function):
    @abstractmethod
    def differentiate(self, domain: np.ndarray) -> np.ndarray:
        """
        Differentiate this function on given ``domain`` and return resulting values.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------*/
# deterministic function

class deterministic(function):
    @util.stack_args(first=1)
    def __call__(self, domain: np.ndarray) -> np.ndarray:
        """
        Evaluate deterministic function on ``domain`` and return function values
        """
        return self.evaluate(domain)

    @abstractmethod
    def evaluate(self, domain: np.ndarray) -> np.ndarray:
        """
        Evaluate this deterministic function with ``domain`` array as input and
        return the resulting values
        """
        raise NotImplementedError

    @abstractmethod
    def differentiate(self, domain: np.ndarray) -> np.ndarray:
        """
        Evaluate the derivative of this deterministic function with ``domain`` array as input and
        return the resulting values
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> np.ndarray:
        """
        Parameters that this deterministic function is parameterized with
        """
        raise NotImplementedError

    @parameters.setter
    @abstractmethod
    def parameters(self, value: np.ndarray) -> None:
        """
        Parameterize this deterministic function with new parameters given as ``value``
        """
        raise NotImplementedError

    @abstractmethod
    def parameters_derivative(self, states: np.ndarray) -> np.ndarray:
        """
        Return a matrix which encodes a transition from current function parameters
        to given ``states``, i.e. f(s) = A * p, where s is the vector
        of ``states``, p is the vector of current parameters,
        and A is the matrix returned.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------*/
# quadratic function

class quadratic(differentiable):
    def __init__(self, parameters: np.ndarray) -> None:
        self._params = np.atleast_2d(parameters)

    def __call__(self, domain: np.ndarray, samples_n: int = 1) -> tuple[np.ndarray] | np.ndarray:
        sample = np.sum(domain.dot(self._params) * domain, axis=1)
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
# linear function

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
# - dynamics

class dynamics(function, uncertainty):
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
# decorator to saturate the output of a function

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
