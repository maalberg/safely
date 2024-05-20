# ---------------------------------------------------------------------------*/
# safe lyapunov (safely) evaluation on a simple 1D example

import numpy as np
import GPy as gpy
from matplotlib import pyplot as plt

import function as fun
import domain as dom
import safe

def plot_dynamics(dyn: fun.uncertain, lyap: safe.lyapunov, safety_thr, ci: float = 2.0) -> None:

    # create figure axes
    fig, axs = plt.subplots(2, 1)

    # format axes
    axs[0].set_title('GP model of uncertain dynamics')
    axs[0].set_xlim(lyap.domain_dims_lim[0, :])
    axs[0].set_ylabel(r'$f(x) + g(x)$')
    axs[1].set_xlim(lyap.domain_dims_lim[0, :])
    axs[1].set_xlabel('$x$')
    axs[1].set_ylabel(r'$\dot{V}(x)$')
    axs[1].set_ylim([-0.5, 0.5])

    # mean and variance of uncertain dynamics
    dyn_mean, dyn_var = dyn.predict(lyap.domain)

    # safety boundary by Lyapunov
    safety_mean, safety_ci = lyap.measure_safety(dyn, ci=2.0)

    # plot the mean of dynamics
    dyn_mean_plt = axs[0].plot(
        lyap.domain, dyn_mean,
        color='red', linestyle='dashed', linewidth=1.5, label=r'mean')

    # plot sample from dynamics
    dyn_samples = dyn.sample(lyap.domain, size=5)
    for sample in dyn_samples:
        dyn_sample_plt = axs[0].plot(
            lyap.domain, sample,
            color='green', alpha=0.75, label=r'dynamics')

    # plot the confidence interval (ci) of dynamics
    dyn_ci = ci * np.sqrt(dyn_var[:, 0])
    axs[0].fill_between(
        lyap.domain[:, 0],
        dyn_mean[:, 0] + dyn_ci, dyn_mean[:, 0] - dyn_ci,
        color='blue', alpha=0.1)

    # plot observed data
    dyn_data = dyn.data_observed
    axs[0].plot(
        dyn_data[0], dyn_data[1], # X and Y
        'x', color='orange', ms=8, mew=2)

    # plot safety
    axs[1].fill_between(
        lyap.domain[:, 0],
        safety_mean + safety_ci, safety_mean - safety_ci,
        color='blue', alpha=0.1)

    # plot sample from Lyapunov
    lyap_samples = lyap.sample(dyn)
    for sample in lyap_samples:
        axs[1].plot(
            lyap.domain, sample)

    # plot safety threshold
    axs[1].plot(*lyap.domain_dims_lim, [safety_thr, safety_thr], 'k-.')

    # plot boundary of the region of attraction
    if np.any(lyap.roa):
        x_safe = lyap.roa_boundary
        y_range = axs[1].get_ylim()
        axs[1].plot([x_safe, x_safe], y_range, 'k-.')
        axs[1].plot([-x_safe, -x_safe], y_range, 'k-.')

    # plot legend
    lns = dyn_mean_plt + dyn_sample_plt
    labels = [l.get_label() for l in lns]
    plt.legend(lns, labels, loc=1, fancybox=True, framealpha=0.75)

    plt.show()


# ---------------------------------------------------------------------------*/
# - preset random number generator to some value to get reproducible results

np.random.seed(8)


# ---------------------------------------------------------------------------*/
# - discretize a domain

dim_lim = [-1, 1]
dim_sz = 1000
domain = dom.gridworld(dim_lim, dim_sz)


# ---------------------------------------------------------------------------*/
# - define linear dynamics

dyn_linear = fun.linearity([-0.25])


# ---------------------------------------------------------------------------*/
# - define uncertain dynamics

kern_var = 0.2**2
kern_lenscale = 0.2
uncertainty_prior = (
    gpy.kern.Matern32(domain.dims_n, lengthscale=kern_lenscale, variance=kern_var) *
    gpy.kern.Linear(domain.dims_n))
uncertainty_var = 0.01 ** 2

dyn = fun.uncertainty(dyn_linear, domain, uncertainty_prior, uncertainty_var)


# ---------------------------------------------------------------------------*/
# - Lyapunov function

# As a Lyapunov function candidate we choose a quadratic function V(x) = x^2, which
# is more generally expressed as V(x) = xT * P * x, where
# P is some positive-definite matrix.
#
lyap_candidate = fun.quadratic(1)
domain_safe_init = np.abs(domain.states.squeeze()) < 0.2
lyap = safe.lyapunov(lyap_candidate, domain, domain_safe_init)


# ---------------------------------------------------------------------------*/
# - Lipschitz constant

# discretization step
tau = np.min(domain.disc)

# Lipschitz constant, test
L = 1

lyap_safe_thr = -tau*L


# ---------------------------------------------------------------------------*/
# - online learning

plot_dynamics(dyn, lyap, lyap_safe_thr)

for i in range(10):
    lyap.update_roa(dyn, safety_thr=lyap_safe_thr)
    lyap.decrease_uncertainty(dyn)

plot_dynamics(dyn, lyap, lyap_safe_thr)
