import numpy as np
import tensorflow as tf

import function as fun
import domain as dom

class lyapunov:
    def __init__(
            self,
            candidate: fun.differentiable, dynamics: fun.stochastic,
            domain: dom.gridworld, domain_safe: list[bool] | None = None) -> None:

        # candidate, dynamics and domain are fixed and cannot be changed later
        self._candidate = candidate
        self._dynamics = dynamics
        self._domain = domain

        self._init_roa(domain_safe)

    @property
    def candidate(self) -> fun.differentiable: return self._candidate

    @property
    def dynamics(self) -> fun.stochastic: return self._dynamics

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
        dyn_mean, dyn_var = self.dynamics.evaluate_error(self.domain.states)

        # lyapunov derivative can indicate whether dynamics decreases toward an equillibrium point
        lyap = self.candidate.differentiate(self.domain.states)

        # so calculate stability error as a lyapunov derivative along dynamics
        err_mean = tf.reduce_sum(lyap * dyn_mean, axis=1, keepdims=True)
        err_var = tf.reduce_sum(lyap**2 * dyn_var, axis=1, keepdims=True)

        return err_mean, err_var

    def find_learnable(self) -> tf.Tensor:
        """
        Find a learnable state inside that is considered safe, but has the most uncertainty.
        """

        # evaluate error from uncertain dynamics
        error = self.dynamics.evaluate_error(self.domain.states)[1]

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
        state_safe = tf.gather(self.domain.states, indices=state_locs, axis=0)

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
        lyap = tf.squeeze(self.candidate(self.domain.states), axis=-1)
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
            state_safe = tf.gather(self.domain.states, indices=safe_locs, axis=0)
            lyap_safe = tf.gather(self.candidate(self.domain.states), indices=safe_locs, axis=0)
            self._roa_bdry = tf.gather(state_safe, indices=tf.argmax(lyap_safe), axis=0)

    def sample(self) -> tf.Tensor:
        dyn_samples = self.dynamics(self.domain.states)[0]
        lyap_der_samples = self.candidate.differentiate(self.domain.states)

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
