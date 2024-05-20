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
# function

class function(metaclass=interface):
    @abstractmethod
    @util.stack_args(first=1)
    def __call__(self, domain: np.ndarray) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Call this function on ``domain`` and return resulting function values,
        potentially returning uncertainty as well
        """
        raise NotImplementedError

    @abstractmethod
    def has_uncertainty(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def dims_n_i(self) -> int:
        """
        Property that holds the number of input dimensions as an integer value
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dims_n_o(self) -> int:
        """
        Property that holds the number of output dimensions as an integer value
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

    def has_uncertainty(self) -> bool:
        return False

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
# uncertain function

class uncertain(function):
    @util.stack_args(first=1)
    def __call__(self, domain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict this uncertain function on given ``domain`` and return the predicted values
        together with corresponding variance
        """
        return self.predict(domain)

    def has_uncertainty(self) -> bool:
        return True

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
