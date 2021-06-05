from typing import Callable, NamedTuple, Tuple

import jax.numpy as jnp


class DualAveragingState(NamedTuple):
    """State carried through the dual averaging procedure.

    log_x
        The logarithm of the current state
    log_x_avg
        The time-weighted average of the values that the logarithm of the state
        has taken so far.
    step
        The current iteration step.
    avg_err
        The time average of the value of the quantity :math:`H_t`, the
        difference between the target acceptance rate and the current
        acceptance rate.
    mu
        Arbitrary point the values of log_step_size are shrunk towards. Chose
        to be :math:`\\log(10 \\epsilon_0)` where :math:`\\epsilon_0` is chosen
        in this context to be the step size given by the
        `find_reasonable_step_size` procedure.

    """

    x: float
    x_avg: float
    step: int
    avg_gradient: float
    mu: float


def dual_averaging(
    t0: int = 10, gamma: float = 0.05, kappa: float = 0.75
) -> Tuple[Callable, Callable]:
    """Find the state that minimizes an objective function using a primal-dual
    subgradient method.

    See [1]_ for a detailed explanation of the algorithm and its mathematical
    properties.

    Parameters
    ----------
    t0: float >= 0
        Free parameter that stabilizes the initial iterations of the algorithm.
        Large values may slow down convergence. Introduced in [2]_ with a default
        value of 10.
    gamma
        Controls the speed of convergence of the scheme. The authors of [2]_ recommend
        a value of 0.05.
    kappa: float in ]0.5, 1]
        Controls the weights of past steps in the current update. The scheme will
        quickly forget earlier step for a small value of `kappa`. Introduced
        in [2]_, with a recommended value of .75

    Returns
    -------
    init
        A function that initializes the state of the dual averaging scheme.
    update
        a function that updates the state of the dual averaging scheme.
    final
        a function that returns the state that minimizes the objective function.

    References
    ----------

    .. [1]: Nesterov, Yurii. "Primal-dual subgradient methods for convex
            problems." Mathematical programming 120.1 (2009): 221-259.
    """

    def init(mu: float = 0) -> DualAveragingState:
        """Initialize the state of the dual averaging scheme.

        The parameter :math:`\\mu` is set to :math:`\\log(10 \\x_init)`
        where :math:`\\x_init` is the initial value of the state.
        """
        step = 1
        avg_gradient: float = 0.0
        x: float = 0.0
        x_avg: float = 0.0
        return DualAveragingState(x, x_avg, step, avg_gradient, mu)

    def update(da_state: DualAveragingState, gradient) -> DualAveragingState:
        """Update the state of the Dual Averaging adaptive algorithm.

        Parameters
        ----------
        gradient:
            The gradient of the function to optimize with respect to the state
            `x`, computed at the current value of `x`.
        da_state:
            The current state of the dual averaging algorithm.

        Returns
        -------
        The updated state of the dual averaging algorithm.
        """
        _, x_avg, step, avg_gradient, mu = da_state
        reg_step = step + t0
        eta_t = step ** (-kappa)
        avg_gradient = (1.0 - (1.0 / (reg_step))) * avg_gradient + gradient / reg_step
        x = mu - (jnp.sqrt(step) / gamma) * avg_gradient
        x_avg = eta_t * x + (1.0 - eta_t) * x_avg
        return DualAveragingState(x, x_avg, step + 1, avg_gradient, mu)

    return init, update
