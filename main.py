from dataclasses import dataclass
from typing import Dict

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


def train_and_eval(model, params, dataset):
    # TODO: What is TrainState doing exactly? Maybe keep it simple instead.
    # TODO: Is there a jax equivalent to torch dataloaders?
    # TODO: Currently not batching. Fix.

    optimizer = optax.adamw(1e-3)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    rngs = {"dropout": jax.random.PRNGKey(0)}

    for epoch in range(5):
        for batch in dataset["train"]:
            batch = {
                "input_ids": jnp.array(batch["input_ids"])[None],
                "labels": jnp.array(batch["labels"])[None],
            }
            state, loss = train_step(state, batch, rngs)
            print(loss)
            #eval_step(state, batch)

#@jax.jit
def train_step(
    state: TrainState,
    batch: Dict[str, jax.Array],
    rngs: Dict[str, jax.random.PRNGKey],
):
    step = state.step
    rngs = {name: jax.random.fold_in(rng, step) for name, rng in rngs.items()}

    def compute_loss(params):
        inputs = batch["input_ids"]
        labels = batch["labels"]
        logits = state.apply_fn(params, inputs, rngs=rngs, train=True)
        label_mask = jax.numpy.where(labels > 0, 1.0, 0.0)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels) * label_mask
        loss = jnp.sum(loss) / jnp.sum(label_mask)
        return loss

    loss, grads = jax.value_and_grad(compute_loss)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def eval_step(state, batch):
    pass


def loss_fn(logits: jax.Array, labels: jax.Array):
    label_mask = jax.numpy.where(labels > 0, 1.0, 0.0)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels) * label_mask
    loss = jnp.sum(loss) / jnp.sum(label_mask)
    return loss


def load_dataset_and_tokenizer(sequence_length: int):
    # TODO: Use another/bigger dataset.
    # TODO: Maybe train the tokenizer on the dataset too.
    # TODO: Use padding instead of skipping last.

    tokenizer = Tokenizer.from_pretrained("distilgpt2")
    dataset = load_dataset("tiny_shakespeare")
    dataset = dataset.map(
        lambda x: {"input_ids": tokenizer.encode(x["text"]).ids},
        remove_columns=["text"],
    )

    chunk_size = sequence_length + 1

    def chunk_input_ids(x):
        assert len(x["input_ids"]) == 1, "Batch size must be 1."
        input_ids = x["input_ids"][0]
        input_ids_chunks = [input_ids[i:i + chunk_size] for i in range(0, len(input_ids), chunk_size)]
        return {"input_ids": input_ids_chunks}

    dataset = dataset.map(chunk_input_ids, batch_size=1, batched=True)
    dataset = dataset.filter(lambda x: len(x["input_ids"]) == chunk_size)
    dataset = dataset.map(lambda x: {"input_ids": x["input_ids"][:-1], "labels": x["input_ids"][1:]})

    return dataset, tokenizer


if __name__ == "__main__":
    rng = jax.random.PRNGKey(123)

    dataset, tokenizer = load_dataset_and_tokenizer(sequence_length=512)

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

    # for epoch in trange(5):
    #     for i in range(10):
    #         input_ids = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=jnp.int32)
    #         labels = jnp.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=jnp.int32)
    #         logits = model.apply(params, rngs=rngs, input_ids=input_ids, train=True)
    #         label_mask = jax.numpy.where(labels > 0, 1.0, 0.0)
    #         loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels) * label_mask
    #         loss = jnp.sum(loss) / jnp.sum(label_mask)
    #         grads = jax.grad(loss)(params)
    #         breakpoint()

    # input_ids = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=jnp.int32)
    # output = model.apply(params, rngs=rngs, input_ids=input_ids, train=True)
    # breakpoint()
