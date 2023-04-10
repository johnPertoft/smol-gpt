from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from datasets import Dataset
from datasets import load_dataset
from einops import rearrange
from flax.training.train_state import TrainState
from tokenizers import Tokenizer
from tqdm import trange


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
        self.wq = nn.Dense(self.config.embed_dim)
        self.wk = nn.Dense(self.config.embed_dim)
        self.wv = nn.Dense(self.config.embed_dim)
        self.wo = nn.Dense(self.config.embed_dim)
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
        # TODO: Add attention mask based on masked input here too, (attention mask).
        mask = self.causal_mask[..., :t, :t]
        attention = jnp.where(mask, attention, -jnp.inf)
        attention = nn.softmax(attention, axis=-1)
        # TODO: Attention dropout here?
        score = attention @ v
        breakpoint()

        """
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
        """


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


def train_and_eval(model, params, dataset):
    optimizer = optax.adamw(1e-3)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )
    breakpoint()

    # TODO: Setup dataloader? Is there an equivalent to torch.utils.data.DataLoader?

    for epoch in range(5):
        for batch in dataset["train"]:
            state = train_step(state, batch)
            #eval_step(state, batch)

#@jax.jit
def train_step(state, batch):
    def compute_loss(params):
        inputs, labels = batch
        logits = state.apply_fn(params, inputs, train=True)
        label_mask = jax.numpy.where(labels > 0, 1.0, 0.0)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels) * label_mask
        loss = jnp.sum(loss) / jnp.sum(label_mask)
        return loss

    loss, grads = jax.value_and_grad(compute_loss)(state.params)
    state = state.apply_gradients(grads=grads)
    return state


def eval_step(state, batch):
    pass


def loss_fn(logits: jax.Array, labels: jax.Array):
    label_mask = jax.numpy.where(labels > 0, 1.0, 0.0)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels) * label_mask
    loss = jnp.sum(loss) / jnp.sum(label_mask)
    return loss


if __name__ == "__main__":
    rng = jax.random.PRNGKey(123)

    # TODO: Train tokenizer too?
    # TODO: Tokenize and prepare dataset.
    tokenizer = Tokenizer.from_pretrained("distilgpt2")
    dataset = load_dataset("tiny_shakespeare")
    #dataset = dataset.map(tokenizer, batched=True)
    #breakpoint()

    config = GPTConfig()
    model = GPT(config)
    params_rng, dropout_rng = jax.random.split(key=rng, num=2)
    rngs = {"params": params_rng, "dropout": dropout_rng}
    params = model.init(
        rngs=rngs,
        input_ids=jnp.empty((1, config.n_positions), dtype=jnp.int32),
        train=True,
    )

    train_and_eval(model, params, dataset)

    for epoch in trange(5):
        for i in range(10):
            input_ids = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=jnp.int32)
            labels = jnp.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=jnp.int32)
            logits = model.apply(params, rngs=rngs, input_ids=input_ids, train=True)
            label_mask = jax.numpy.where(labels > 0, 1.0, 0.0)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels) * label_mask
            loss = jnp.sum(loss) / jnp.sum(label_mask)
            grads = jax.grad(loss)(params)
            breakpoint()

    input_ids = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=jnp.int32)
    output = model.apply(params, rngs=rngs, input_ids=input_ids, train=True)
    breakpoint()
