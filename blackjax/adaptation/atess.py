"""Adaptive Transport Elliptical Slice sampler"""

from typing import Callable

import jax
import jax.numpy as jnp
import optax

import blackjax.adaptation.chain_adaptation as chain_adaptation
from blackjax.types import PyTree


def base(
    kernel_factory,
    optim,
    loss,
    num_batch: int,
    batch_size: int,
    n_iter: int = 10,
    eca: bool = True,
    batch_fn: Callable = jax.pmap,
):
    def parameter_gn(batch_state, current_iter, param, state):
        batch_position = batch_state.position
        param_state, loss_value = optimize(
            param,
            state,
            loss,
            optim,
            n_iter,
            batch_position,
        )
        return param_state

    if eca:
        init, update = chain_adaptation.parallel_eca(
            kernel_factory, parameter_gn, num_batch, batch_size, batch_fn
        )
    else:
        init, update = chain_adaptation.cross_chain(
            kernel_factory, parameter_gn, num_batch * batch_size, batch_fn
        )

    def final(last_state: chain_adaptation.ChainState, param_state: PyTree) -> PyTree:
        if eca:
            return None
        param_state = parameter_gn(
            last_state.states,
            last_state.current_iter,
            *param_state,
        )
        return kernel_factory(*param_state)

    return init, update, final


def optimize(param, state, loss, optim, n_iter, positions):
    def step_fn(carry, i):
        params, opt_state = carry
        loss_value, grads = jax.value_and_grad(loss)(params, positions)
        updates, opt_state_ = optim.update(grads, opt_state, params)
        params_ = optax.apply_updates(params, updates)
        return jax.lax.cond(
            jnp.isfinite(loss_value)
            & jnp.isfinite(jax.flatten_util.ravel_pytree(grads)[0]).all(),
            lambda _: ((params_, opt_state_), loss_value),
            lambda _: ((params, opt_state), jnp.nan),
            None,
        )

    param_state, loss_value = jax.lax.scan(step_fn, (param, state), jnp.arange(n_iter))
    return param_state, loss_value
