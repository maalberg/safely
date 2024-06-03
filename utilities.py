import numpy as np
import scipy as sp

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