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
    def __call__(self, domain: np.ndarray, samples_n: int = 1) -> tuple[np.ndarray]:
        """
        Take ``samples_n`` number of function samples with ``domain`` as input and
        return these samples in a tuple.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_error(self, domain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate function error in given ``domain`` and return a predicted mean value
        together with a corresponding variance.
        """
        raise NotImplementedError

    @abstractmethod
    def observe_datapoints(self, domain: np.ndarray, value: np.ndarray) -> None:
        """
        Let function observe datapoints in ``domain`` with given ``value``.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dims_n_i(self) -> int:
        """
        Number of input dimensions.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dims_n_o(self) -> int:
        """
        Number of output dimensions.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def datapoints_observed(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Datapoints observed by this function, see method ``observe_datapoints``.
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

    def evaluate_error(self, domain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass

    def observe_datapoints(self, domain: np.ndarray, value: np.ndarray) -> None:
        pass

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

    @property
    def datapoints_observed(self) -> tuple[np.ndarray, np.ndarray] | None:
        pass


# ---------------------------------------------------------------------------*/
# uncertain function

class uncertain(function):
    @util.stack_args(first=1)
    def __call__(self, domain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict this uncertain function on given ``domain`` and return the predicted values
        together with corresponding variance
        """
        return self.predict(domain)

    def evaluate_error(self, domain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass

    def observe_datapoints(self, domain: np.ndarray, value: np.ndarray) -> None:
        pass

    @abstractmethod
    def sample(self, domain: np.ndarray, size: int = 1) -> np.ndarray:
        """
        Sample this uncertain function with ``domain`` array as input and return
        resulting samples. This method draws ``size`` samples from a normal
        distribution. This method suits when there is a need to plot
        the underlying function.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, domain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict the uncertainty of this function on given ``domain`` and
        return the resulting mean and variance
        """
        raise NotImplementedError

    @abstractmethod
    def decrease_uncertainty(self, domain: np.ndarray, value: np.ndarray) -> None:
        """
        Decrease the uncertainty of this uncertain function by letting the function observe
        the given ``value`` on certain ``domain``
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dim_o_active(self) -> int:
        """
        Property that holds the active output dimension of this uncertain function as an integer value

        Note that this makes the uncertain function scalar.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def data_observed(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Property that holds (X, Y) data that has been observed by this uncertain function in order to decrease uncertainty,
        see `decrease_uncertainty`
        """
        raise NotImplementedError

    @property
    def datapoints_observed(self) -> tuple[np.ndarray, np.ndarray] | None:
        pass


# ---------------------------------------------------------------------------*/
# quadratic function

class quadratic(deterministic):
    def __init__(self, cost: np.ndarray) -> None:
        self._impl_parameters = np.atleast_2d(cost)

    def evaluate(self, domain: np.ndarray):
        return np.sum(domain.dot(self._impl_parameters) * domain, axis=1)

    def differentiate(self, domain):
        return 2 * domain.dot(self._impl_parameters)

    def parameters_derivative(self, domain: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def parameters(self) -> np.ndarray:
        return self._impl_parameters

    @parameters.setter
    def parameters(self, value) -> None:
        self._impl_parameters = value

    @property
    def dims_n_i(self) -> int:
        return np.shape(self._impl_parameters)[1]

    @property
    def dims_n_o(self) -> int:
        return np.shape(self._impl_parameters)[0]


# ---------------------------------------------------------------------------*/
# uncertainty modeling in terms of a Gaussian process

class uncertainty(uncertain):
    def __init__(
            self,
            apriori: deterministic, apriori_domain: dom.gridworld,
            uncertainty: gpy.kern.Kern, uncertainty_var: float,
            dim_o_active: int = 0) -> None:

        self._impl_dims_n_i = apriori.dims_n_i
        self._impl_dims_n_o = apriori.dims_n_o

        self._impl_dim_o_active = dim_o_active

        # use apriori dynamics as the mean function of a Gaussian process
        gp_mean = gpy.core.Mapping(apriori.dims_n_i, apriori.dims_n_o)
        gp_mean.f = apriori
        gp_mean.update_gradients = lambda a, b: None

        # a gaussian process with initial observed data at the origin with X=0, Y=0
        self._impl_gp = gpy.core.GP(
            np.zeros((1, apriori.dims_n_i)), np.zeros((1, apriori.dims_n_o)),
            uncertainty,
            gpy.likelihoods.Gaussian(variance=uncertainty_var),
            mean_function=gp_mean)

        self._impl_uncertainty_var = uncertainty_var

        # based on the domain of apriori dynamics, construct a less discretized domain for fast sampling
        self._impl_sampling_X = dom.gridworld(apriori_domain.dims_lim, 100).states
        self._impl_sampling_alpha = None

    def sample(self, domain: np.ndarray, size: int = 1) -> np.ndarray:

        # # add mean function values to the sample
        # if self._impl_gp.mean_function is not None:
        #     value += self._impl_gp.mean_function.f(domain)

        samples = self._impl_gp.posterior_samples(domain, size=size)
    
        # format samples as a tuple of samples
        return [samples[..., this_sample] for this_sample in range(samples.shape[-1])]

    def predict(self, domain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._impl_gp.predict_noiseless(domain)

    def decrease_uncertainty(self, state: np.ndarray, value: np.ndarray) -> None:
        # update the observed data of the internal Gaussian process
        self._impl_gp.set_XY(
            np.row_stack([self._impl_gp.X, state]),
            np.row_stack([self._impl_gp.Y, value]))

    @property
    def dims_n_i(self) -> int:
        return self._impl_dims_n_i

    @property
    def dims_n_o(self) -> int:
        return self._impl_dims_n_o

    @property
    def dim_o_active(self) -> int:
        return self._impl_dim_o_active
    
    @property
    def data_observed(self) -> tuple[np.ndarray, np.ndarray]:
        return (self._impl_gp.X, self._impl_gp.Y)


# ---------------------------------------------------------------------------*/
# linearity

class linearity(deterministic):
    def __init__(self, matrices: list[np.ndarray]) -> None:
        self._impl_dynamics = np.column_stack(tuple(map(np.atleast_2d, matrices)))

    @util.stack_args(first=1)
    def __call__(self, domain: np.ndarray) -> np.ndarray:
        """
        Evaluate linear function derivative on ``domain`` and return a tuple with derivatives
        and zero uncertainty
        """
        return self.differentiate(domain)

    def evaluate(self, domain: np.ndarray) -> np.ndarray:
        """
        Implement in case there is a need to evaluate y = C*x,
        where C is output matrix, x is state and y is an observed system variable.
        """
        raise NotImplementedError

    def differentiate(self, domain: np.ndarray) -> np.ndarray:
        return domain.dot(self._impl_dynamics.T)

    @property
    def parameters(self) -> np.ndarray:
        return self._impl_dynamics

    @parameters.setter
    def parameters(self, value) -> None:
        self._impl_dynamics = value

    def parameters_derivative(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def dims_n_i(self) -> int:
        return np.shape(self._impl_dynamics)[1]

    @property
    def dims_n_o(self) -> int:
        return np.shape(self._impl_dynamics)[0]


# ---------------------------------------------------------------------------*/
# - dynamics

class dynamics(function):
    def __init__(
            self,
            model: function, error: gpy.kern.Kern = None,
            policy: function = None) -> None:

        self._dims_n_i = model.dims_n_i
        self._dims_n_o = model.dims_n_o

        # if error is present, construct stochastic dynamics,
        # otherwise the dynamics are deterministic
        if error is not None:
            # use model as the mean function of a Gaussian process
            gp_mean = gpy.core.Mapping(model.dims_n_i, model.dims_n_o)
            gp_mean.f = model
            gp_mean.update_gradients = lambda a, b: None

            # a gaussian process with initial observed data at the origin with x=0, y=0
            gp = gpy.core.GP(
                np.zeros((1, model.dims_n_i)), np.zeros((1, model.dims_n_o)),
                error,
                gpy.likelihoods.Gaussian(variance=0), # no observaion noise at the moment
                mean_function=gp_mean)

            # define a sampling method for stochastic dynamics
            def sample(domain: np.ndarray, samples_n: int) -> tuple[np.ndarray]:
                samples = gp.posterior_samples(domain, size=samples_n)

                # format samples as a tuple of samples
                return [samples[..., this_sample] for this_sample in range(samples.shape[-1])]

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

                return [sample for this in range(samples_n)]

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

        self.policy = policy

    def __call__(self, domain: np.ndarray, samples_n: int = 1) -> np.ndarray:

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
    def dims_n_i(self) -> int:
        return self._dims_n_i

    @property
    def dims_n_o(self) -> int:
        return self._dims_n_o

    @property
    def datapoints_observed(self) -> tuple[np.ndarray, np.ndarray]:
        # return observed datapoints, if any
        return self._observations()


# ---------------------------------------------------------------------------*/
# - inverted pendulum

class pendulum_inv(deterministic):
    def __init__(
            self,
            mass: float, length: float, friction: float,
            state_action_max: tuple[list, list] = None) -> None:
        self.mass = mass
        self.length = length
        self.friction = friction
        self.state_action_max = state_action_max

    @util.stack_args(first=1)
    def __call__(self, domain: np.ndarray) -> np.ndarray:
        return self.differentiate(domain)

    def evaluate(self, domain: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def differentiate(self, domain: np.ndarray) -> np.ndarray:
        """
        Take time-derivative of pendulum dynamics by solving an ordinary differential equation,
        where ``domain`` contains arrays of states and corresponding actions.
        Every row of input arrays represents data to evaluate,
        whereas columns represent dimensions.
        """
        state, action = np.split(domain, indices_or_sections=[2], axis=1)
        state, action = self.denormalize(state, action)

        gravity = self.gravity
        length = self.length
        friction = self.friction
        inertia = self.inertia

        # physical dynamics
        angle, angular_velocity = np.split(state, indices_or_sections=2, axis=1)
        angular_acceleration = gravity / length * np.sin(angle) + action / inertia
        if friction > 0:
            angular_acceleration -= friction / inertia * angular_velocity

        state_derivative = np.column_stack((angular_velocity, angular_acceleration))
        state_derivative = self.normalize_state(state_derivative)
        return state_derivative

    @property
    def parameters(self) -> np.ndarray:
        raise NotImplementedError

    @parameters.setter
    def parameters(self, value) -> None:
        raise NotImplementedError

    def parameters_derivative(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def dims_n_i(self) -> int:
        return 3

    @property
    def dims_n_o(self) -> int:
        return 2

    @property
    def inertia(self):
        return self.mass * self.length ** 2

    @property
    def gravity(self):
        return 9.81

    def _impl_get_state_denorm(self) -> np.ndarray:
        return np.diag(np.atleast_1d(self.state_action_max[0]))

    def _impl_get_state_norm(self) -> np.ndarray:
        return np.diag(np.diag(self._impl_get_state_denorm()) ** -1)

    def _impl_get_action_denorm(self) -> np.ndarray:
        return np.diag(np.atleast_1d(self.state_action_max[1]))

    def _impl_get_action_norm(self) -> np.ndarray:
        return np.diag(np.diag(self._impl_get_action_denorm()) ** -1)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        if self.state_action_max is None:
            return state

        return state.dot(self._impl_get_state_norm())

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        if self.state_action_max is None:
            return action

        return action.dot(self._impl_get_action_norm())

    def normalize(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.normalize_state(state), self.normalize_action(action)

    def denormalize_state(self, state: np.ndarray) -> np.ndarray:
        if self.state_action_max is None:
            return state

        return state.dot(self._impl_get_state_denorm())

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        if self.state_action_max is None:
            return action

        return action.dot(self._impl_get_action_denorm())

    def denormalize(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.denormalize_state(state), self.denormalize_action(action)

    def linearize(self) -> tuple[np.ndarray, np.ndarray]:
        gravity = self.gravity
        length = self.length
        friction = self.friction
        inertia = self.inertia

        # linearized dynamics, where sinx = x
        a = np.array([
            [0, 1],
            [gravity / length, -friction / inertia]])

        # action input
        b = np.array([
            [0],
            [1 / inertia]])

        # provided the maximum values of states and actions are available,
        # normalize linearized matrices, adhering to the following signal scheme
        #
        # normalized output <- normalize * matrix * denormalize <- normalized input
        if self.state_action_max is not None:
            state_norm = self._impl_get_state_norm()

            a = np.linalg.multi_dot((state_norm, a, self._impl_get_state_denorm()))
            b = np.linalg.multi_dot((state_norm, b, self._impl_get_action_denorm()))

        return a, b

class dlqr(deterministic):
    def __init__(self, a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray) -> None:
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
        self._impl_control = np.linalg.solve(bpb, bpa)
        self._impl_parameters = p

    def evaluate(self, domain: np.ndarray) -> np.ndarray:
        domain = np.asarray(domain)
        return -domain.dot(self._impl_control.T)

    def differentiate(self, domain: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters_derivative(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def parameters(self) -> np.ndarray:
        return self._impl_parameters

    @parameters.setter
    def parameters(self, value) -> None:
        self._impl_parameters = value

    @property
    def dims_n_i(self) -> int:
        return np.shape(self._impl_control)[1]

    @property
    def dims_n_o(self) -> int:
        return np.shape(self._impl_control)[0]


# ---------------------------------------------------------------------------*/
# function decorator to saturate the output of a deterministic function

class saturated(deterministic):
    def __init__(self, func: deterministic, clipping: float) -> None:
        self._impl_func = func
        self._impl_clipping = clipping

    def evaluate(self, domain: np.ndarray) -> np.ndarray:
        value = self._impl_func.evaluate(domain)
        return np.clip(value, -self._impl_clipping, self._impl_clipping)

    def differentiate(self, domain: np.ndarray) -> np.ndarray:
        return self._impl_func.differentiate(domain)

    @property
    def parameters(self) -> np.ndarray:
        return self._impl_func.parameters

    @parameters.setter
    def parameters(self, value) -> None:
        self._impl_func.parameters = value

    def parameters_derivative(self, states: np.ndarray) -> np.ndarray:
        return self._impl_func.parameters_derivative(states)

    @property
    def dims_n_i(self) -> int:
        return self._impl_func.dims_n_i

    @property
    def dims_n_o(self) -> int:
        return self._impl_func.dims_n_o
