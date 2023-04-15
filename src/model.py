from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange



@dataclass
class GPTConfig:
    n_layers: int = 6
    n_heads: int = 8
    embed_dim: int = 512
    dropout: float = 0.1
    vocab_size: int = 50304
    n_positions: int = 2048


class GPT(nn.Module):
    config: GPTConfig

    def setup(self):
        self.input_embeddings = nn.Embed(self.config.vocab_size, self.config.embed_dim)
        self.position_embeddings = nn.Embed(self.config.n_positions, self.config.embed_dim)
        self.blocks = [GPTBlock(self.config) for _ in range(self.config.n_layers)]
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False)

    def __call__(self, input_ids: jax.Array, *, train: bool):
        pos_ids = jnp.arange(input_ids.shape[1]).reshape((1, -1))
        x = self.input_embeddings(input_ids) + self.position_embeddings(pos_ids)
        for block in self.blocks:
            x = block(x, train=train)
        logits = self.lm_head(x)
        # TODO: There's some dropout and normalization missing here.
        return logits


class GPTBlock(nn.Module):
    config: GPTConfig

    def setup(self):
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
        self.attention = CausalSelfAttention(self.config)
        self.mlp = MLP(self.config)

    def __call__(self, x: jax.Array, *, train: bool):
        x = x + self.attention(self.ln1(x), train=train)
        x = x + self.mlp(self.ln2(x), train=train)
        return x


class CausalSelfAttention(nn.Module):
    config: GPTConfig

    def setup(self):
        assert self.config.embed_dim % self.config.n_heads == 0, "Incompatible number of heads and embedding dimensions."
        self.wq = nn.Dense(self.config.embed_dim, use_bias=False)
        self.wk = nn.Dense(self.config.embed_dim, use_bias=False)
        self.wv = nn.Dense(self.config.embed_dim, use_bias=False)
        self.wo = nn.Dense(self.config.embed_dim, use_bias=False)
        # TODO: Can this be kept elsewhere instead?
        causal_mask = jnp.tril(jnp.ones((self.config.n_positions, self.config.n_positions), dtype="bool"))
        causal_mask = rearrange(causal_mask, "i j -> 1 1 i j")
        self.causal_mask = causal_mask

    def __call__(self, x: jax.Array, *, train: bool):
        # Sequence length.
        t = x.shape[1]

        # Compute q, k, v projections.
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Split into heads.
        q = rearrange(q, "b l (h d) -> b l h d", h=self.config.n_heads)
        k = rearrange(k, "b l (h d) -> b l h d", h=self.config.n_heads)
        v = rearrange(v, "b l (h d) -> b l h d", h=self.config.n_heads)

        # TODO: Rearrange in one step in the above instead?
        # TODO: Or just use einsum?
        q = rearrange(q, "b l h d -> b h l d")
        k = rearrange(k, "b l h d -> b h l d")
        v = rearrange(v, "b l h d -> b h l d")

        # Compute attention.
        scaling = (1.0 / jnp.sqrt(k.shape[-1]))
        attention = q @ k.transpose(0, 1, 3, 2) * scaling
        # TODO: Add mask to attention for padded values etc.
        mask = self.causal_mask[..., :t, :t]
        attention = jnp.where(mask, attention, -jnp.inf)
        attention = nn.softmax(attention, axis=-1)
        # TODO: Attention dropout here?
        output = attention @ v
        
        # Merge heads.
        output = rearrange(output, "b h l d -> b l (h d)")

        # Compute output projection.
        output = self.wo(output)
        # TODO: Residual dropout here?

        return output


class MLP(nn.Module):
    config: GPTConfig

    def setup(self):
        self.w1 = nn.Dense(self.config.embed_dim * 4)
        self.act = nn.gelu
        self.w2 = nn.Dense(self.config.embed_dim)
        self.dropout = nn.Dropout(self.config.dropout)

    def __call__(self, x: jax.Array, *, train: bool):
        x = self.w1(x)
        x = self.act(x)
        x = self.w2(x)
        x = self.dropout(x, deterministic=not train)
        return x

