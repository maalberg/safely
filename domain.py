import numpy as np
import tensorflow as tf
import gpflow

import utilities as utils

class gridworld:
    def __init__(self, dims_lim: list[tuple], dims_sz: list[int]) -> None:

        self._dims_lim = tf.experimental.numpy.atleast_2d(dims_lim)
        self._dims_n = len(self._dims_lim)
        self._dims_sz = tf.cast(
            tf.broadcast_to(dims_sz, shape=[self._dims_n]),
            dtype=tf.int64)

        # demarcate grid axes as evenly spaced intervals,
        # and since the size of dimensions may be different,
        # it is important to keep the axes as a list, not as a numpy array
        self._axes = [
            tf.linspace(start, stop, n)
            for (start, stop), n in zip(self._dims_lim, self._dims_sz)
        ]

        # derive discretization steps along each axis
        self._step = tf.stack([axis[1] - axis[0] for axis in self._axes])

        # create grid points
        mesh = tf.meshgrid(*self._axes, indexing='ij') # asterisk unpacks axes into axes[0], axes[1], etc.
        self._points = tf.concat(
            list(tf.reshape(col, shape=[-1, 1]) for col in mesh), # reshape into column vectors
            axis=1)

    @property
    def dims_n(self) -> int: return self._dims_n

    @property
    def dims_lim(self) -> tf.Tensor: return self._dims_lim

    @property
    def dims_sz(self) -> tf.Tensor: return self._dims_sz

    @property
    def rectangles_n(self) -> int: return tf.math.reduce_prod(self.dims_sz - 1)

    @property
    def offset(self) -> tf.Tensor: return self.dims_lim[:, 0]

    @property
    def points(self) -> tf.Tensor: return self._points

    @property
    def step(self) -> tf.Tensor: return self._step

    @property
    def dims_axes(self) -> tf.Tensor: return self._axes

    def __len__(self) -> int: return tf.math.reduce_prod(self.dims_sz)

    def get_points(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Return points at given indices

        It is possible to compute the points based on the regularity of the
        domain grid, i.e. without seeing/knowing the actual data.
        """

        # arrange indices, such that every row is going to accomodate a point
        indices = tf.experimental.numpy.atleast_1d(indices)
        indices = tf.transpose(
            tf.concat(
                tf.unravel_index(indices, self.dims_sz),
                axis=0))

        # convert indices to points
        return tf.cast(indices, dtype=gpflow.default_float()) * self.step + self.offset

    def locate_points(self, points: tf.Tensor) -> tf.Tensor:
        """
        Return indices corresponding to given ``points``.
        """
        points = tf.experimental.numpy.atleast_2d(points)

        # calculate locations as integers
        locations = (points - self.offset) * (1. / self.step)
        # round float numbers to their nearest integers
        locations = tf.cast(tf.math.rint(locations), dtype=tf.int64)

        # convert a tuple of index arrays - note the transpose - into an array of flat indices
        return utils.tf_ravel_multi_index(
            tf.transpose(locations),
            dims=self.dims_sz)

    def shift_points(self, points: tf.Tensor, needs_clipping=True) -> tf.Tensor:
        """
        Shift ``points`` to a new domain [0, point_max - domain_offset]
        """
        points = tf.experimental.numpy.atleast_2d(points)
        points_shifted = points - self.offset

        if needs_clipping:
            # clip shifted states to the boundaries of the new domain
            clip_min = tf.zeros_like(self.dims_lim[:, 0])
            clip_max = self.dims_lim[:, 1] - self.offset
            clipping = tf.concat([
                tf.expand_dims(clip_min, axis=-1), # tensorflow forbids 1D arrays to be stacked as columns
                tf.expand_dims(clip_max, axis=-1)], axis=1)

            eps = np.finfo(float).eps # clip precision is machine-specific

            points_shifted = tf.clip_by_value(
                points_shifted,
                clip_value_min=clipping[:, 0] + 2 * eps,
                clip_value_max=clipping[:, 1] - 2 * eps)

        return points_shifted

    def locate_rectangles(self, points: tf.Tensor) -> tf.Tensor:
        """
        Locate rectangles that are closest to given ``points`` and
        return the indices of these rectangles.
        """
        points = tf.experimental.numpy.atleast_2d(points)
        rectangles = []

        for this, (axis, size) in enumerate(zip(self.dims_axes, self.dims_sz)):
            # locate bins [their indices in fact] on domain axes where the given points reside
            states_binning = tf.cast(
                tf.keras.ops.digitize(points[:, this], axis) - 1,
                dtype=tf.int64)

            # clip out-of-range bins
            states_binning = tf.clip_by_value(
                states_binning, clip_value_min=0, clip_value_max=size - 2)

            rectangles.append(states_binning)
        return utils.tf_ravel_multi_index(rectangles, self.dims_sz - 1)

    def locate_origins(self, rectangles: tf.Tensor) -> tf.Tensor:
        """
        Given the indices of ``rectangles``, locate their upper-left corners, or origins.
        """

        # The indices of rectangles are passed as linear indices.
        # In case of a 1D array this is simple, as the second element will have index 1, for example.
        # In case of more dimensions, one needs to wrap the indices in some way,
        # for example row-by-row like in Python, starting from the upper-left
        # position. So given `dims` parameter, function unravel_index
        # goes from linear indices to n-dimensional (wrapped) ones.
        #
        # Now using the fact that an n-by-n grid has (n-1)-by-(n-1) rectangles,
        # which is also true when n != m, we specify that `dims` parameter
        # as n - 1. The function then returns the index of a point
        # in ND format, which lies at the upper-left
        # corner of the given rectangle.
        nd_indices = tf.concat(
            tf.unravel_index(indices=rectangles, dims=self.dims_sz - 1),
            axis=0)

        nd_indices = tf.experimental.numpy.atleast_2d(nd_indices)

        # And ravel_multi_index function simply reverts back the unravelled ND index back
        # to its linear format. By setting dims parameter back to n we locate
        # that upper-left rectangle corner in the original setting.
        return utils.tf_ravel_multi_index(nd_indices, dims=self.dims_sz)

    def sample_continuous(self, samples_n: int) -> tf.Tensor:
        """
        Generate ``samples_n`` continuous samples from this domain. Here the notion
        of continuous means that the samples are not bound to the grid of
        the domain, but are freely allocated throughout the domain.
        """

        # generate random numbers to scale the ranges of domain dimensions
        scaling = tf.random.uniform(
            minval=0, maxval=1,
            shape=(samples_n, self.dims_n),
            dtype=gpflow.default_float())

        # scale the ranges and add the result to the starting point of these ranges
        return scaling * tf.transpose(tf.experimental.numpy.diff(self.dims_lim, axis=1)) + self.offset
