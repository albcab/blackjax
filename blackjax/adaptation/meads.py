"""Public API for Maximum-Eigenvalue Adaptation of Damping and Step-size kernel"""

from typing import Callable

import jax
import jax.numpy as jnp

import blackjax.adaptation.chain_adaptation as chain_adaptation
from blackjax.types import PyTree


def base(
    kernel_factory: Callable,
    logprob_grad_fn: Callable,
    num_batch: int,
    batch_size: int,
    eca: bool = True,
    batch_fn: Callable = jax.vmap,
):
    def maximum_eigenvalue(matrix: PyTree):
        X = jnp.vstack(
            [leaf.reshape(leaf.shape[0], -1).T for leaf in jax.tree_leaves(matrix)]
        ).T
        n, _ = X.shape
        S = X @ X.T
        diag_S = jnp.diag(S)
        lamda = jnp.sum(diag_S) / n
        lamda_sq = (jnp.sum(S**2) - jnp.sum(diag_S**2)) / (n * (n - 1))
        return lamda_sq / lamda

    def parameter_gn(batch_state, current_iter):
        batch_position = batch_state.position
        mean_position = jax.tree_map(lambda p: p.mean(axis=0), batch_position)
        sd_position = jax.tree_map(lambda p: p.std(axis=0), batch_position)
        batch_norm = jax.tree_map(
            lambda p, mu, sd: (p - mu) / sd,
            batch_position,
            mean_position,
            sd_position,
        )
        batch_grad = batch_fn(logprob_grad_fn)(batch_position)
        batch_grad_scaled = jax.tree_map(
            lambda grad, sd: grad * sd, batch_grad, sd_position
        )
        epsilon = jnp.minimum(
            0.5 / jnp.sqrt(maximum_eigenvalue(batch_grad_scaled)), 1.0
        )
        gamma = jnp.maximum(
            1.0 / jnp.sqrt(maximum_eigenvalue(batch_norm)),
            1.0 / ((current_iter + 1) * epsilon),
        )
        alpha = 1.0 - jnp.exp(-2.0 * epsilon * gamma)
        delta = alpha / 2
        step_size = jax.tree_map(lambda sd: epsilon * sd, sd_position)
        return step_size, alpha, delta

    if eca:
        init, update = chain_adaptation.parallel_eca(
            kernel_factory, parameter_gn, num_batch, batch_size, batch_fn
        )
    else:
        init, update = chain_adaptation.cross_chain(
            kernel_factory, parameter_gn, num_batch * batch_size, batch_fn
        )

    def final(last_state: chain_adaptation.ChainState) -> PyTree:
        if eca:
            return None
        parameters = parameter_gn(
            last_state.states,
            last_state.current_iter,
        )
        return kernel_factory(*parameters)

    return init, update, final