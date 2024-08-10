import tensorflow as tf
import gpflow

from collections import namedtuple

import function as fun
import domain as dom


# ---------------------------------------------------------------------------*/
# - safety boundary inside a state domain

domain_boundary = namedtuple('domain_boundary', 'min max')


# ---------------------------------------------------------------------------*/
# - safety certificate

class certificate:
    def __init__(
            self,
            domain: dom.gridworld, safe: list[bool], rules: fun.lyapunov,
            threshold: float = tf.constant(0, dtype=gpflow.default_float())) -> None:

        self._dom = domain
        self._lya = rules

        self.threshold = threshold

        self._set_safety(safe)

        self.update()

    @property
    def domain(self) -> dom.gridworld: return self._dom

    @property
    def rules(self) -> fun.lyapunov: return self._lya

    def safe(self, state: tf.Tensor = None) -> bool | list[bool]:
        """
        Query if a given ``state`` can be certified as safe. If no ``state`` is given,
        then return the safety status of the whole domain.
        """
        return self._safety if state is None else tf.gather(self._safety, indices=self.domain.locate_points(state))

    @property
    def safe_boundary(self) -> domain_boundary:
        """
        Return the boundary of the current domain region that is certified as safe.
        """
        states = self.domain.points

        # locations [in fact indices] where states are safe
        safe_locs = tf.squeeze(tf.where(self.safe()), axis=-1)

        # extract minimum and maximum safe states
        states_safe = tf.gather(states, indices=safe_locs, axis=0)
        return domain_boundary(tf.reduce_min(states_safe), tf.reduce_max(states_safe))

    @property
    def safe_uncertainty(self) -> tf.Tensor:
        """
        Return a state that is certified as safe, but has the most uncertainty (variance).
        """

        states = self.domain.points

        # evaluate state error from uncertain dynamics
        error = self.rules.dynamics.error(states).variance

        # get state locations that are certified as safe
        #
        # Certificate is a one-dimensional array of bool values, so method tf.where
        # will return a two-dimensional array of indices with size (n, 1),
        # where n is the number of true elements. The last dimension
        # is then squeezed, or removed, because the bool's are
        # meant to show state locations, regardless of the
        # state dimensionality. And states are allocated
        # along the first dimension, i.e. axis = 0.
        safe_locs = tf.squeeze(tf.where(self.safe()), axis=-1)

        # gather all states and errors that are considered safe
        #
        # Based on a one-dimensional certified locations, tf.gather will return slices
        # from states and errors with the same dimensionality.
        error_safe = tf.gather(error, indices=safe_locs, axis=0)
        state_safe = tf.gather(states, indices=safe_locs, axis=0)

        # gather a certified state, but with maximum error (uncertainty)
        #
        # Method tf.argmax is expected to return a location as a list, not
        # as a single scalar. In this case, tf.gather will gather
        # the corresponding state with proper dimensions.
        return tf.gather(state_safe, indices=tf.argmax(error_safe), axis=0)

    def update(self) -> None:
        """
        Update certificate.
        """

        states = self.domain.points

        # evaluate lyapunov error and derive the upper bound of its confidence interval
        err_mean, err_var = self._lya.error(states)
        err_ci = 2.0 * tf.sqrt(err_var)
        err_ci_u = err_mean + err_ci

        # certificate is valid,
        # when the upper bound of a lyapunov error is less than some safety threshold or
        # when there is a user-defined certificate for a specific domain region
        cert = tf.logical_or(tf.squeeze(err_ci_u < self.threshold, axis=-1), self.safe())

        # determine initial region of attraction, i.e. a region in which a state
        # becomes attracted to an equillibrium point; the region is
        # defined in terms of a lyapunov function value.
        #
        # roa is squeezed to remove the last dimension in order to comply with
        # the rest of the code, so e.g. if roa is of size (1000, 1),
        # then the squeezed version is of size (1000).
        roa = tf.squeeze(self._lya.candidate(states), axis=-1)
        roa_max = tf.reduce_max(roa)
        roa_accuracy = roa_max / 1e10

        # update initial region of attraction by searching the true boundary of roa,
        # inside which the roa is totally safe, i.e. all states inside the
        # region have the attraction property and are, thus, safe.
        #
        # The boundary is searched on a roa interval by applying a safety constraint.
        safety_cstr = lambda roa_bdry : tf.reduce_all(cert[roa <= roa_bdry])
        roa_interval = [0, roa_max + roa_accuracy]
        roa_boundary = self._search_roa_boundary(roa_interval, roa_accuracy, safety_cstr)

        # in case a boundary is found, update current certificate
        if roa_boundary is not None: self._set_safety(roa <= roa_boundary)

    def _set_safety(self, safe: list[bool]) -> None:
        """
        Specify where the current domain is ``safe``.
        """
        self._safety = safe

    def _search_roa_boundary(
            self,
            interval: list[float, float], accuracy: float,
            constraint) -> float | None:
        """
        Find the upper boundary of a region of attraction inside a specified ``interval``.
        During the search the points inside ``interval`` are checked for safety
        using a callable ``constraint``. The search continues
        until ``accuracy`` is reached. If the whole
        region is unsafe, the method returns None.
        """
        if not constraint(interval[0]):
            # the starting point immediately does not satisfy safety,
            # so return nothing
            return None

        if constraint(interval[1]):
            # the end point satisfies safety, so entire region is safe,
            # and thus the end can be returned as boundary
            return interval[1]

        # apply binary search algorithm to find boundary until accuracy is reached
        while interval[1] - interval[0] > accuracy:
            mean = (interval[0] + interval[1]) / 2
            if constraint(mean): interval[0] = mean
            else: interval[1] = mean

        # return the upper bound of located interval
        return interval[1]

class lyapunov:
    def __init__(
            self,
            candidate: fun.function, dynamics: fun.dynamics,
            domain: dom.gridworld, domain_safe: list[bool] | None = None) -> None:

        # candidate, dynamics and domain are fixed and cannot be changed later
        self._candidate = candidate
        self._dynamics = dynamics
        self._domain = domain

        self._init_roa(domain_safe)

    @property
    def candidate(self) -> fun.function: return self._candidate

    @property
    def dynamics(self) -> fun.dynamics: return self._dynamics

    @property
    def domain(self) -> dom.gridworld: return self._domain

    @property
    def domain_safe(self) -> list[bool]: return self._roa

    @property
    def domain_safe_bdry(self) -> float | None: return self._roa_bdry

    def evaluate_error(self) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Evaluate error in dynamical stability and return the mean and variance of this error.
        The error is evaluated in terms of lyapunov derivative along dynamics.
        """

        # evaluate error of dynamics
        dyn_mean, dyn_var = self.dynamics.error(self.domain.points)

        # lyapunov derivative can indicate whether dynamics decreases toward an equillibrium point
        lyap = self.candidate.gradient(self.domain.points)

        # so calculate stability error as a lyapunov derivative along dynamics
        err_mean = tf.reduce_sum(lyap * dyn_mean, axis=1, keepdims=True)
        err_var = tf.reduce_sum(lyap**2 * dyn_var, axis=1, keepdims=True)

        return err_mean, err_var

    def find_learnable(self) -> tf.Tensor:
        """
        Find a learnable state inside that is considered safe, but has the most uncertainty.
        """

        # evaluate error from uncertain dynamics
        error = self.dynamics.evaluate_error(self.domain.points)[1]

        # get state locations that are considered safe
        #
        # Region of attraction is a one-dimensional array of bool values, so method tf.where
        # will return a two-dimensional array of indices with size (n, 1),
        # where n is the number of true elements. The last dimension
        # is then squeezed, or removed, because the bool's are
        # meant to show state locations, regardless of the
        # state dimensionality. And states are allocated
        # along the first dimension, i.e. axis = 0.
        state_locs = tf.squeeze(tf.where(self.domain_safe), axis=-1)

        # gather all state errors that are considered safe
        #
        # Based on a one-dimensional state locations, tf.gather will return slices
        # from states and errors with the same state dimensionality.
        error_safe = tf.gather(error, indices=state_locs, axis=0)
        state_safe = tf.gather(self.domain.points, indices=state_locs, axis=0)

        # gather the currently learnable state
        #
        # Method tf.argmax is expected to return a location as a list, not
        # as a single scalar. In this case, tf.gather will gather
        # the corresponding state with proper dimensions.
        return tf.gather(state_safe, indices=tf.argmax(error_safe), axis=0)

    def update_safety(self, threshold: float | list[float] = 0.) -> None:

        # evaluate dynamics error and derive the upper bound of its confidence interval
        err_mean, err_var = self.evaluate_error()
        err_ci = 2.0 * tf.sqrt(err_var)
        err_ci_u = err_mean + err_ci

        # domain is considered safe, when the upper bound of dynamics error is less
        # than a specific, user-defined, safety threshold;
        # the domain is then augmented with safety
        # information derived earlier
        safe = tf.logical_or(tf.squeeze(err_ci_u < threshold, axis=-1), self.domain_safe)

        # determine initial region of attraction, i.e. a region in which a state
        # becomes attracted to an equillibrium point; the region is
        # defined in terms of a lyapunov function value.
        #
        # roa is squeezed to remove the last dimension in order to comply with
        # the rest of the code, so e.g. if roa is of size (1000, 1),
        # then the squeezed version is of size (1000).
        lyap = tf.squeeze(self.candidate(self.domain.points), axis=-1)
        lyap_max = tf.reduce_max(lyap)
        lyap_acc = lyap_max / 1e10

        # update initial region of attraction by searching the true boundary of roa,
        # inside which the roa is totally safe, i.e. all states inside the
        # region have the attraction property and are, thus, safe
        safe_cstr = lambda lyap_bdry : tf.reduce_all(safe[lyap <= lyap_bdry])
        lyap_search = [0, lyap_max + lyap_acc]
        lyap_bdry = self._find_lyap_boundary(lyap_search, lyap_acc, safe_cstr)

        # in case boundary is found, update current region of attraction
        if lyap_bdry is not None:
            self._roa = lyap <= lyap_bdry

            # determine the boundary of a region of attraction
            safe_locs = tf.squeeze(tf.where(self.domain_safe), axis=-1)
            state_safe = tf.gather(self.domain.points, indices=safe_locs, axis=0)
            lyap_safe = tf.gather(self.candidate(self.domain.points), indices=safe_locs, axis=0)
            self._roa_bdry = tf.gather(state_safe, indices=tf.argmax(lyap_safe), axis=0)

    def sample(self) -> tf.Tensor:
        dyn_samples = self.dynamics(self.domain.points)[0]
        lyap_der_samples = self.candidate.gradient(self.domain.points)

        return tf.reduce_sum(lyap_der_samples * dyn_samples, axis=1, keepdims=True)

    def _find_lyap_boundary(self, lyap: list[float, float], lyap_acc: float, safe_cstr) -> float | None:
        """
        Find the upper boundary of a region of attraction inside an interval specified by ``lyap``.
        During the search the points inside ``lyap`` are safety constrained
        using callable ``safe_cstr``. The search continues
        until accuracy ``lyap_acc`` is reached.
        If the whole region is unsafe,
        the method returns None.
        """
        if not safe_cstr(lyap[0]):
            # the starting point immediately does not satisfy safety,
            # so return nothing
            return None

        if safe_cstr(lyap[1]):
            # the end point satisfies safety, so entire region is safe,
            # and thus the end can be returned as boundary
            return lyap[1]

        # apply binary search algorithm to find boundary until accuracy is reached
        while lyap[1] - lyap[0] > lyap_acc:
            mean = (lyap[0] + lyap[1]) / 2
            if safe_cstr(mean): lyap[0] = mean
            else: lyap[1] = mean

        # return the upper bound of located interval
        return lyap[1]

    def _init_roa(self, domain_safe: list[bool]) -> None:
        self._roa = domain_safe
        self._roa_initial = domain_safe
        self._roa_bdry = None
        if domain_safe is not None:
            self.update_safety()
        else:
            self._roa = tf.zeros(len(self.domain), dtype=tf.bool)
