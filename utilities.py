import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


# ---------------------------------------------------------------------------*/
# - synthesize a digital linear-quadratic regulator

def make_dlqr(a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesize a digital linear-quadratic regulator (lqr) using matrices ``a``, ``b``, ``q`` and ``r``,
    where ``a`` and ``b`` are the system and input matrices of a plant to be regulated,
    and where ``q`` and ``r`` are the state and control cost matrices, respectively.
    The method returns a tuple with the regulator matrix K to be used as u = -K*x,
    where x is the full state of the plant; and the solution to an algebraic
    Riccati equation, which was used during the synthesis of K matrix.
    """
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
    control = np.linalg.solve(bpb, bpa)
    solution = p

    return control, solution


# ---------------------------------------------------------------------------*/
# - one-dimensional delaunay triangulation

class delaunay1d:
    """
    Class that mimics scipy delaunay algorithm for the triangulation of 1D data
    """
    def __init__(self, points: np.ndarray) -> None:
        if points.shape[1] > 1:
            raise AttributeError('err > this class is designed for 1D inputs')
        if points.shape[0] > 2:
            raise AttributeError('err > this class supports only two points')

        self.points = points

        # there will be only one simplex
        self.nsimplex = len(points) - 1

        self._bound_min = np.min(points)
        self._bound_max = np.max(points)

        # indices of the points forming the simplices in this triangulation
        #
        # shape of simplices is (nsimplex, ndim + 1),
        # see the documentation of scipy delaunay algorithm
        #
        # the following single row represents one simplex formed by two points
        self.simplices = np.array([[0, 1]])

    def find_simplex(self, points: np.ndarray):
        """
        Find simplices containing given points
        """
        points = points.squeeze()
        bound_miss = points > self._bound_max
        bound_miss |= points < self._bound_min

        return np.where(bound_miss, -1, 0)


# ---------------------------------------------------------------------------*/
# - plot triangulation in three-dimensions

def plot3d_triangulation(tri):
    parameters = tri.parameters
    states = tri._domain.states
    simplices = tri.get_simplices(np.arange(tri.nsimplex))

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    ax.plot_trisurf(
        states[:, 0], states[:, 1], parameters[:, 0],
        triangles=simplices)

    return fig, ax
