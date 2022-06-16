"""Public API for the Generalized (Non-reversible w/ persistent momentum) HMC Kernel"""
from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import blackjax.mcmc.hmc as hmc
import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
import blackjax.mcmc.proposal as proposal
from blackjax.types import PRNGKey, PyTree

__all__ = ["GHMCState", "init", "kernel"]


class GHMCState(NamedTuple):
    position: PyTree
    momentum: PyTree
    potential_energy: float
    potential_energy_grad: PyTree
    slice: float


def init(
    rng_key: PRNGKey,
    position: PyTree,
    logprob_fn: Callable,
    logprob_grad_fn: Optional[Callable] = None,
):
    def potential_fn(x):
        return -logprob_fn(x)

    if logprob_grad_fn:
        potential_energy_grad = jax.tree_map(
            lambda g: -1.0 * g, logprob_grad_fn(position)
        )
        potential_energy = potential_fn(position)

    else:
        potential_energy, potential_energy_grad = jax.value_and_grad(potential_fn)(
            position
        )

    p, unravel_fn = ravel_pytree(position)
    key_mometum, key_slice = jax.random.split(rng_key)
    momentum = unravel_fn(jax.random.normal(key_mometum, p.shape))
    slice = jax.random.uniform(key_slice, minval=-1.0, maxval=1.0)

    return GHMCState(position, momentum, potential_energy, potential_energy_grad, slice)


def kernel(
    noise_gn: Callable = lambda _: 0.0,
    divergence_threshold: float = 1000,
):

    _, kinetic_energy_fn, _ = metrics.gaussian_euclidean(jnp.ones(1))
    sample_proposal = proposal.nonreversible_slice_sampling

    def one_step(
        rng_key: PRNGKey,
        state: GHMCState,
        logprob_fn: Callable,
        step_size: PyTree,  # float,
        alpha: float,
        delta: float,
        logprob_grad_fn: Optional[Callable] = None,
    ) -> Tuple[GHMCState, hmc.HMCInfo]:
        def potential_fn(x):
            return -logprob_fn(x)

        symplectic_integrator = velocity_verlet(
            potential_fn, kinetic_energy_fn, logprob_grad_fn
        )
        proposal_generator = hmc.hmc_proposal(
            symplectic_integrator,
            kinetic_energy_fn,
            step_size,
            divergence_threshold=divergence_threshold,
            sample_proposal=sample_proposal,
        )

        key_momentum, key_noise = jax.random.split(rng_key)
        position, momentum, potential_energy, potential_energy_grad, slice = state
        # New momentum is persistent
        momentum = update_momentum(key_momentum, position, momentum, alpha)
        # Slice is non-reversible
        slice = ((slice + 1.0 + delta + noise_gn(key_noise)) % 2) - 1.0

        integrator_state = integrators.IntegratorState(
            position, momentum, potential_energy, potential_energy_grad
        )
        proposal, info = proposal_generator(slice, integrator_state)
        proposal = hmc.flip_momentum(proposal)
        state = GHMCState(
            proposal.position,
            proposal.momentum,
            proposal.potential_energy,
            proposal.potential_energy_grad,
            info.acceptance_probability,
        )

        return state, info

    return one_step


def velocity_verlet(
    potential_fn: Callable,
    kinetic_energy_fn: integrators.EuclideanKineticEnergy,
    logprob_grad_fn: Optional[Callable] = None,
) -> integrators.EuclideanIntegrator:

    if logprob_grad_fn:
        potential_and_grad_fn = lambda x: (
            potential_fn(x),
            jax.tree_map(lambda g: -1.0 * g, logprob_grad_fn(x)),
        )
    else:
        potential_and_grad_fn = jax.value_and_grad(potential_fn)
    kinetic_energy_grad_fn = jax.grad(kinetic_energy_fn)

    def one_step(
        state: integrators.IntegratorState, step_size: PyTree
    ) -> integrators.IntegratorState:
        position, momentum, _, potential_energy_grad = state

        momentum = jax.tree_util.tree_map(
            lambda momentum, potential_grad, step_size: momentum
            - 0.5 * step_size * potential_grad,
            momentum,
            potential_energy_grad,
            step_size,
        )

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = jax.tree_util.tree_map(
            lambda position, kinetic_grad, step_size: position
            + step_size * kinetic_grad,
            position,
            kinetic_grad,
            step_size,
        )

        potential_energy, potential_energy_grad = potential_and_grad_fn(position)
        momentum = jax.tree_util.tree_map(
            lambda momentum, potential_grad, step_size: momentum
            - 0.5 * step_size * potential_grad,
            momentum,
            potential_energy_grad,
            step_size,
        )

        return integrators.IntegratorState(
            position,
            momentum,
            potential_energy,
            potential_energy_grad,
        )

    return one_step


def update_momentum(rng_key, position, momentum, alpha):
    m, _ = ravel_pytree(momentum)
    momentum_generator, *_ = metrics.gaussian_euclidean(
        1 / alpha * jnp.ones(jnp.shape(m))
    )
    momentum = jax.tree_map(
        lambda prev_momentum, shifted_momentum: prev_momentum * jnp.sqrt(1.0 - alpha)
        + shifted_momentum,
        momentum,
        momentum_generator(rng_key, position),
    )

    return momentum
