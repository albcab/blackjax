"""Public API for ChEES-HMC"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import optax

import blackjax.mcmc.hmc as hmc
import blackjax.mcmc.integrators as integrators 
import blackjax.mcmc.proposal as proposal
import blackjax.adaptation.chain_adaptation as chain_adaptation
from blackjax.adaptation.step_size import dual_averaging_adaptation
from blackjax.types import Array


class HarmonicAcceptance(NamedTuple):
    acceptance_probability: float


class ChESSState(NamedTuple):
    state: hmc.HMCState
    current_iter: int


def base(
    kernel_factory: Callable,
    optim,
    num_chain: int,
    halton_sequence: Array,
    batch_fn: Callable = jax.vmap,
):
    def parameter_gn(
        batch_state, current_iter, 
        batch_info, initial_batch_state,
        parameters, moving_averages,
        halton_number, optim_state,
    ):
        step_size, trajectory_length = parameters
        step_size_ma, trajectory_length_ma = moving_averages

        harmonic_mean = 1. / jnp.mean(1. / batch_info.acceptance_probability)
        da_init, da_update, da_final = dual_averaging_adaptation(target=0.651)
        da_state = da_init(step_size)
        info = HarmonicAcceptance(harmonic_mean)
        new_step_size = da_final(da_update(da_state, info))
        new_step_size_ma = 0.9 * step_size_ma + 0.1 * new_step_size

        proposals_mean = jax.tree_map(lambda p: p.mean(axis=0), batch_info.proposal.state.position)
        initials_mean = jax.tree_map(lambda p: p.mean(axis=0), initial_batch_state.states.state.position)
        proposals_centered = jax.tree_map(lambda p, pm: p - pm, batch_info.proposal.state.position, proposals_mean)
        initials_centered = jax.tree_map(lambda p, pm: p - pm, initial_batch_state.states.state.position, initials_mean)

        proposals_matrix = batch_fn(lambda p: ravel_pytree(p)[0])(proposals_centered)
        initials_matrix = batch_fn(lambda p: ravel_pytree(p)[0])(initials_centered)
        momentums_matrix = batch_fn(lambda m: ravel_pytree(m)[0])(batch_info.proposal.state.momentum)

        trajectory_gradients = (
            halton_sequence[current_iter-1] * trajectory_length * (
                # jnp.einsum('ij,ij->i', proposals_matrix, proposals_matrix)
                batch_fn(lambda p: jnp.dot(p, p))(proposals_matrix)
                # - jnp.einsum('ij,ij->i', initials_matrix, initials_matrix)
                - batch_fn(lambda p: jnp.dot(p, p))(initials_matrix)
            # ) * jnp.einsum('ij,ij->i', proposals_matrix, momentums_matrix)
            ) * batch_fn(lambda p, m: jnp.dot(p, m))(initials_matrix, momentums_matrix)
        )
        trajectory_gradient = jnp.sum(
            batch_info.acceptance_probability * trajectory_gradients
        ) / jnp.sum(batch_info.acceptance_probability)

        log_trajectory_length = jnp.log(trajectory_length)
        updates, new_optim_state = optim.update(trajectory_gradient, optim_state, log_trajectory_length)
        new_log_trajectory_length = optax.apply_updates(log_trajectory_length, updates)
        new_trajectory_length = jnp.exp(new_log_trajectory_length)
        new_trajectory_length_ma = 0.9 * trajectory_length_ma + 0.1 * new_trajectory_length

        return (
            (new_step_size, new_trajectory_length), 
            (new_step_size_ma, new_trajectory_length_ma),  
            halton_sequence[current_iter], new_optim_state,
        )

    init, update = chain_adaptation.cross_chain(
        kernel_factory, parameter_gn, num_chain, batch_fn
    )

    def extended_init(initial_states, positions):
        empty_pytree = jax.tree_map(lambda p: jnp.zeros(p.shape), positions)
        exact_acceptance = 0.651 * jnp.ones(num_chain) 
        empty_float = jnp.zeros(num_chain, dtype=float)
        empty_bools = jnp.zeros(num_chain, dtype=bool)
        init_integrator = integrators.IntegratorState(
            empty_pytree, empty_pytree,
            empty_float, empty_pytree,
        )
        init_proposal = proposal.Proposal(
            init_integrator, empty_float,
            empty_float, empty_float,
        )
        init_infos = hmc.HMCInfo(
            empty_pytree, exact_acceptance,
            empty_bools, empty_bools,
            empty_float, init_proposal,
            empty_float,
        )
        init_state = ChESSState(hmc.HMCState(
            empty_pytree, empty_float,
            empty_pytree
        ), jnp.zeros(num_chain, dtype=int))
        return init(initial_states), init_infos, init(init_state)

    return extended_init, update