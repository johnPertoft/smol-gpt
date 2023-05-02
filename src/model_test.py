import jax
import jax.numpy as jnp
import optax
from einops import rearrange

from .model import GPT
from .model import GPTConfig
from .model import SelfAttention


def test_self_attention_gradient_causality():    
    c = GPTConfig()
    m = SelfAttention(c)
    causal_mask = jnp.tril(jnp.ones((c.n_positions, c.n_positions), dtype="bool"))
    causal_mask = rearrange(causal_mask, "i j -> 1 1 i j")
    params = m.init(
        rngs={"params": jax.random.PRNGKey(1), "dropout": jax.random.PRNGKey(1)},
        x=jnp.ones((1, 16, c.embed_dim), dtype=jnp.float32),
        causal_mask=causal_mask,
        train=True,
    )
    
    # Compute the gradients between all outputs and inputs and assert that outputs
    # don't depend on inputs from the future. We run this in eval mode (train=False)
    # so that the dropout layer don't interfere.
    f = lambda x: m.apply(params, x, causal_mask=causal_mask, train=False)
    g = jax.jacfwd(f)
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 16, c.embed_dim))
    grads = g(x)
    grads = rearrange(grads, "1 t1 d1 1 t2 d2 -> t1 d1 t2 d2")
    abs_grads_sum = jnp.abs(grads).sum(axis=(1, 3))  # Sum gradients along embedding dim at each timestep.
    assert jnp.allclose(abs_grads_sum, jnp.tril(abs_grads_sum))
