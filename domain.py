import numpy as np

class gridworld:
    def __init__(self, dims_lim: list[tuple], dims_sz: list[int]) -> None:

        self._impl_dims_lim = np.atleast_2d(dims_lim)
        self._impl_dims_n = len(self._impl_dims_lim)

        # derive grid axes demarcated with discrete points (ticks on these axes)
        self._impl_dims_sz = np.broadcast_to(dims_sz, self._impl_dims_n)
        self._impl_axes = np.atleast_2d([
            np.linspace(lo, hi, n)
            for (lo, hi), n in zip(self._impl_dims_lim, self._impl_dims_sz)
        ])

        # get discretization along each axis
        self._impl_disc = np.diff(self._impl_axes, axis=1)[:, 0]

        # initialize grid points property to none
        self._impl_grid_points = None

    @property
    def dims_n(self) -> int:
        return self._impl_dims_n

    @property
    def dims_lim(self) -> np.ndarray:
        return self._impl_dims_lim

    @property
    def dims_sz(self) -> np.ndarray:
        return self._impl_dims_sz

    @property
    def rectangles_n(self) -> int:
        return np.prod(self.dims_sz - 1)

    @property
    def offset(self) -> np.ndarray:
        return self.dims_lim[:, 0]

    @property
    def states(self) -> np.ndarray:
        if self._impl_grid_points is None:
            mesh = np.meshgrid(*self._impl_axes, indexing='ij') # asterisk stands for argument unpacking
            self._impl_grid_points = np.column_stack(list(col.ravel() for col in mesh)) # ravel returns a flattened view of an array, not a copy

        return self._impl_grid_points

    @property
    def disc(self) -> float:
        """
        Property holding the discretization step of this domain as a float value
        """
        return self._impl_disc

    @property
    def dims_axes(self) -> np.ndarray:
        return self._impl_axes

    def __len__(self) -> int:
        return np.prod(self.dims_sz)

    def get_states(self, indices: np.ndarray) -> np.ndarray:
        """
        Return states at given indices

        It is possible to compute the states based on the regularity of the
        domain grid, i.e. without seeing/knowing the actual data.
        """

        # arrange indices, such that every row is going to accomodate a state
        indices = np.atleast_1d(indices)
        indices = np.row_stack(np.unravel_index(indices, self.dims_sz)).T

        # convert indices to states
        return indices * self.disc + self.offset

    def locate_states(self, states: np.ndarray) -> np.ndarray:
        """
        Return indices corresponding to given states
        """
        states = np.atleast_2d(states)

        # calculate locations as integers
        locations = (states - self.offset) * (1. / self.disc)
        locations = np.rint(locations).astype(int) # round float numbers to their nearest integers

        # convert a tuple of index arrays - note the transpose - into an array of flat indices
        return np.ravel_multi_index(locations.T, self.dims_sz)

    def shift_states(self, states: np.ndarray, needs_clipping=True) -> np.ndarray:
        """
        Shift ``states`` to a new domain [0, state_max - domain_offset]
        """
        states = np.atleast_2d(states)
        states_shifted = states - self.offset

        if needs_clipping:
            # clip shifted states to the boundaries of the new domain
            dom_shifted = np.column_stack([
                np.zeros_like(self.dims_lim[:, 0]),
                self.dims_lim[:, 1] - self.offset])
            eps = np.finfo(float).eps # clip precision is machine-specific
            np.clip(
                states_shifted,
                dom_shifted[:, 0] + 2 * eps,
                dom_shifted[:, 1] - 2 * eps,
                out=states_shifted)

        return states_shifted

    def locate_rectangles(self, states: np.ndarray) -> np.ndarray:
        """
        Locate rectangles that are closest to given ``states`` and
        return the indices of these rectangles
        """
        states = np.atleast_2d(states)
        rectangles = []

        for this, (axis, size) in enumerate(zip(self.dims_axes, self.dims_sz)):
            # locate bins [their indices in fact] on domain axes where given states reside
            states_binning = np.digitize(states[:, this], axis) - 1

            # clip out-of-range bins
            np.clip(states_binning, 0, size - 2, out=states_binning)

            rectangles.append(states_binning)
        return np.ravel_multi_index(rectangles, self.dims_sz - 1)

    def locate_origins(self, rectangles: np.ndarray) -> np.ndarray:
        """
        Given the indices of ``rectangles``, locate the upper-left corners,
        or origins, of these ``rectangles``
        """

        # The indices of rectangles are passed as linear indices.
        # In case of a 1D array this is simple, as the second element will have index 1, for example.
        # In case of more dimensions, one needs to wrap the indices in some way,
        # for example row-by-row like in Python, starting from the upper-left
        # position. So given shape parameter, function unravel_index
        # goes from linear indices to n-dimensional (wrapped) ones.
        #
        # Now using the fact that an n-by-n grid has (n-1)-by-(n-1) rectangles,
        # which is also true when n != m, we specify that shape parameter
        # as n - 1. The function then returns the index of a point
        # in ND format, which lies at the upper-left
        # corner of the given rectangle.
        nd_indices = np.row_stack(np.unravel_index(rectangles, shape=self.dims_sz - 1))

        # And ravel_multi_index function simply reverts back the unravelled ND index back
        # to its linear format. By setting dims parameter back to n we locate
        # that upper-left rectangle corner in the original setting.
        return np.ravel_multi_index(np.atleast_2d(nd_indices), dims=self.dims_sz)
