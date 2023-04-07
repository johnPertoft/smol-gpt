from dataclasses import dataclass
from typing import Optional

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


@dataclass
class GPTOutput:
    pass


class GPT(nn.Module):
    config: GPTConfig

    def setup(self):
        # TODO: GPT uses just regular embeddings for positions I think?
        self.embeddings = nn.Embed(self.config.vocab_size, self.config.embed_dim)
        self.blocks = [GPTBlock(self.config) for _ in range(self.config.n_layers)]

    def __call__(self, input_ids: jax.Array, *, train: bool, targets: Optional[jax.Array] = None):
        pass


class GPTBlock(nn.Module):
    config: GPTConfig

    def setup(self):
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
        self.attention = CausalSelfAttention(self.config)
        self.mlp = MLP(self.config)

    def __call__(self, x: jax.Array, *, train: bool):
        # TODO
        pass


class CausalSelfAttention(nn.Module):
    config: GPTConfig

    def setup(self):
        assert self.config.embed_dim % self.config.n_heads == 0, "Incompatible number of heads and embedding dimensions."
        self.wq = nn.Dense(self.config.embed_dim)
        self.wk = nn.Dense(self.config.embed_dim)
        self.wv = nn.Dense(self.config.embed_dim)
        self.wo = nn.Dense(self.config.embed_dim)

    def __call__(self, x: jax.Array, *, train: bool):
        # Compute q, k, v projections.
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Split into heads.
        q = rearrange(q, "b l (h d) -> b l h d", h=self.config.n_heads)
        k = rearrange(k, "b l (h d) -> b l h d", h=self.config.n_heads)
        v = rearrange(v, "b l (h d) -> b l h d", h=self.config.n_heads)

        # Compute attention.
        # TODO


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


if __name__ == "__main__":
    print("hello world")
