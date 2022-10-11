"""Public API for NUTS adaptation w/ dual averaging"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

import blackjax.mcmc.nuts as nuts
import blackjax.mcmc.integrators as integrators 
import blackjax.adaptation.chain_adaptation as chain_adaptation
from blackjax.adaptation.step_size import dual_averaging_adaptation


class HarmonicAcceptance(NamedTuple):
    acceptance_probability: float


def base(
    kernel_factory: Callable,
    num_chain: int,
    batch_fn: Callable = jax.vmap,
):
    def parameter_gn(
        batch_state, current_iter, 
        batch_info, step_size,
    ):
        harmonic_mean = 1. / jnp.mean(1. / batch_info.acceptance_probability)
        da_init, da_update, da_final = dual_averaging_adaptation(target=0.8)
        da_state = da_init(step_size)
        info = HarmonicAcceptance(harmonic_mean)
        new_step_size = da_final(da_update(da_state, info))

        return (new_step_size,)

    init, update = chain_adaptation.cross_chain(
        kernel_factory, parameter_gn, num_chain, batch_fn
    )

    def extended_init(initial_states, positions):
        empty_pytree = jax.tree_map(lambda p: jnp.zeros(p.shape), positions)
        exact_acceptance = 0.8 * jnp.ones(num_chain) 
        empty_float = jnp.zeros(num_chain, dtype=float)
        empty_bools = jnp.zeros(num_chain, dtype=bool)
        empty_int = jnp.ones(num_chain, dtype=int)
        init_integrator = integrators.IntegratorState(
            empty_pytree, empty_pytree,
            empty_float, empty_pytree,
        )
        init_infos = nuts.NUTSInfo(
            empty_pytree, empty_bools, empty_bools,
            empty_float, init_integrator, init_integrator,
            empty_int, empty_int, exact_acceptance,
        )
        return init(initial_states), init_infos

    return extended_init, update