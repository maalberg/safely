import numpy as np
import function as fun
import domain as dom

class lyapunov:
    def __init__(
            self,
            candidate: fun.deterministic, domain: dom.gridworld,
            roa_init: list[bool] = None) -> None:

        self.candidate = candidate

        self._impl_domain = domain
        self._impl_roa_init = roa_init

        self._impl_reset_roa(domain, roa_init)

    def differentiate_along(self, tragectory: np.ndarray, squared: bool = False) -> np.ndarray:
        """
        Calculate the derivative of this Lyapunov function along dynamical system ``tragectory``
        and return the resulting derivative. The ``tragectory`` can also represent
        variance of an uncertain system. In this case, ``squared`` must be set to True.
        """
        lyap_derivative = self.candidate.differentiate(self._impl_domain.states)
        if squared: lyap_derivative = lyap_derivative**2
        return np.sum(lyap_derivative * tragectory, axis=1)

    def sample(self, dynamics: fun.function, samples_n: int = 1) -> np.ndarray:
        samples = dynamics(self._impl_domain.states, samples_n=samples_n)

        return [self.differentiate_along(sample) for sample in samples]

    def measure_safety(
            self,
            dynamics: fun.function,
            ci: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
        """
        This method calculates the boundaries of the derivative of a Lyapunov function candidate
        along the trajectories of ``dynamics``. The result returned is then the mean and
        confidence interval (CI) of Lyapunov-based stability. Optional ``control``
        appends the evaluation of ``dynamics`` with actions; of course the
        ``dynamics`` must be ready to accept these actions. The
        optional parameter ``ci`` can be used to adjust the
        confidence interval; its default value of 1.96
        sorresponds to 95% interval for normal data.
        """

        # prepare states and actions
        states = self._impl_domain.states
        #actions = control(states) if control is not None else None

        # evaluate dynamics
        mean, var = dynamics.evaluate_error(states)

        # calculate Lyapunov derivative along the mean and variance of dynamics
        lyap_d = self.candidate.differentiate(states)
        lyap_dmean = np.sum(lyap_d * mean, axis=1)
        lyap_dvar = np.sum(lyap_d**2 * var, axis=1)

        # as the returned upper bound calculate an upper confidence interval (ci)
        lyap_dci = ci * np.sqrt(lyap_dvar)
        return lyap_dmean, lyap_dci

    def update_roa(
            self,
            dynamics: fun.function,
            safety_thr: float | list[float] = 0, needs_reset: bool = False) -> None:
        """
        Update the current region of attraction (ROA) based on ``dynamics`` and a ``safety_thr``.
        There is a possibility to reset the ROA with a ``needs_reset`` flag.
        """

        if needs_reset is True:
            self._impl_reset_roa(self._impl_domain, self._impl_roa_init)

        # -------------------------------------------------------------------*/
        # based on a Lyapunov derivative, find a safe set inside the domain

        # measure safety boundry
        safety_mean, safety_ci = self.measure_safety(dynamics)
        safety_bdry = safety_mean + safety_ci

        # a safe set is where the Lyapunov derivative is less than a threshold or
        # the safety is explicitly specified by the user
        safe = safety_bdry < safety_thr
        safe = np.logical_or(safe, self._impl_roa)

        # -----------------------------------------------------------------*/
        # find the maximum boundary c of a region of attraction curlyV,
        # so that whenever a state trajectory lands inside the
        # region of attraction, the trajectory stays in this
        # region and eventually converges to origin

        # introduce an interval (0, Vmax] of Lyapunov values V(x),
        # where we want to find the maximum boundary c,
        # such that V(x) <= c for c > 0
        lyap = self.candidate.evaluate(self._impl_domain.states)
        lyap_max = np.max(lyap)
        search_accuracy = lyap_max / 1e10
        search_interval = [0, lyap_max + search_accuracy]

        # constraint which defines the region of attraction, i.e. all points inside
        # a Lyapunov surface V(x) <= c have a negative-definite derivative
        roa_cstr = lambda c : np.all(safe[lyap <= c])
        search_interval = self._impl_search_roa_boundary(search_interval, roa_cstr, search_accuracy)

        # -----------------------------------------------------------------*/
        # update the current region of attraction

        if search_interval is not None:
            self._impl_roa[:] = lyap <= search_interval[0]

    def _impl_reset_roa(self, domain: dom.gridworld, roa_init: list[bool] = None) -> None:
        self._impl_roa = np.zeros(np.prod(domain.dims_sz), dtype=bool)
        if roa_init is not None: self._impl_roa[roa_init] = True

    def _impl_search_roa_boundary(self, search_interval, search_constraint, search_accuracy):
        """
        Search the boundary of a region of attraction on `search_interval` by applying
        `search_constraint` to the points of `search_interval`
        until `search_accuracy` is reached

        The method uses a binary search algrithm to find the level.
        """
        if not search_constraint(search_interval[0]):
            return None

        if search_constraint(search_interval[1]):
            search_interval[0] = search_interval[1]
            return search_interval

        while search_interval[1] - search_interval[0] > search_accuracy:
            mean = (search_interval[0] + search_interval[1]) / 2

            if search_constraint(mean):
                search_interval[0] = mean
            else:
                search_interval[1] = mean

        return search_interval

    def find_uncertainty(self, dynamics: fun.dynamics) -> np.ndarray:
        """
        Find a state with maximum uncertainty which is still safe to sample given the ``dynamics``
        """

        # determine a location (index) in a safe set, where the given uncertain function
        # exhibits maximum uncertainty
        _, error = dynamics.evaluate_error(self._impl_domain.states)
        this_error_max = np.argmax(error[self._impl_roa, 0]) # fix me <- active dimension

        # based on location, extract the corresponding state
        state_uncertain = self._impl_domain.states[ # in all domain
            self._impl_roa][                # narrow down to a safe set
                [this_error_max], :].copy()     # and extract a row vector denoting a (multidimensional) state

        return state_uncertain

    def decrease_uncertainty(self, dynamics: fun.dynamics) -> None:
        """
        Decrease uncertainty of ``dynamics`` by letting it sample a safe state which has maximum uncertainty
        """
        state_uncertain = self.find_uncertainty(dynamics)
        dynamics.observe_datapoints(state_uncertain, dynamics(state_uncertain)[0])

    @property
    def roa_boundary(self) -> float:
        domain = self.domain
        roa = self.roa
        value = self.candidate.evaluate(domain)
        return domain[roa][np.argmax(value[roa])]

    @property
    def roa(self) -> list[bool]:
        return self._impl_roa

    @property
    def domain(self) -> np.ndarray:
        return self._impl_domain.states

    @property
    def domain_dims_lim(self) -> np.ndarray:
        return self._impl_domain.dims_lim
