from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import jax
import jax.numpy as jnp
import optax
from datasets import Dataset
from flax.training.train_state import TrainState

from .data import get_dataset_and_tokenizer
from .model import GPT
from .model import GPTConfig


def train_and_eval():
    # TODO:
    # - Write this with more low level contructs? I.e. without TrainState.
    # - Use a dataloader of some sort?
    # - Write this with multi gpu + multi host support? for fun
    # - Add a learning rate scheduler
    
    rng = jax.random.PRNGKey(123)

    dataset, tokenizer = get_dataset_and_tokenizer(256)

    config = GPTConfig()
    model = GPT(config)
    params_rng, dropout_rng = jax.random.split(key=rng)
    rngs = {"params": params_rng, "dropout": dropout_rng}
    params = model.init(
        rngs=rngs,
        input_ids=jnp.empty((1, config.n_positions), dtype=jnp.int32),
        train=True,
    )

    optimizer = optax.adamw(1e-3)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    for epoch in range(5):
        train_data_loader = create_data_loader(dataset["train"], batch_size=4, rng=rng)
        for batch in train_data_loader:
            state, loss = train_step(state, batch, rng)
            print(f"Epoch {epoch + 1:02} Step {state.step:04} - Loss: {loss}")

        eval_data_loader = create_data_loader(dataset["validation"], batch_size=4, rng=rng)
        for batch in eval_data_loader:
            # TODO
            pass


def create_data_loader(dataset: Dataset, batch_size: int, rng: Optional[jax.random.PRNGKey] = None):
    # TODO:
    # - Maybe add padding/truncation.
    # - Maybe add BoS/EoS tokens etc.

    if rng is not None:
        indices = jax.random.permutation(rng, len(dataset))
    else:
        indices = jnp.arange(len(dataset))
    num_batches = len(dataset) // batch_size  # Skip incomplete batch.
    indices = indices[: num_batches * batch_size]
    indices = indices.reshape((num_batches, batch_size))
    for idxs in indices:
        batch = dataset[idxs]
        batch = {k: jnp.array(v) for k, v in batch.items()}
        yield batch


#@jax.jit
def train_step(state: TrainState, batch: Dict[str, Any], rng: jax.random.PRNGKey) -> Tuple[TrainState, float]:
    rng = jax.random.fold_in(rng, state.step)

    def compute_loss(params, inputs, labels):
        logits = state.apply_fn(params, inputs, train=True, rngs={"dropout": rng})
        loss = loss_fn(logits, labels)
        return loss

    inputs = batch["input_ids"]
    labels = batch["labels"]
    loss, grads = jax.value_and_grad(compute_loss)(state.params, inputs, labels)
    state = state.apply_gradients(grads=grads)
    return state, loss
    
    


#@jax.jit
def eval_step(params, batch: Dict[str, Any]):
    pass


def loss_fn(logits, labels):
    label_mask = jax.numpy.where(labels > 0, 1.0, 0.0)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels) * label_mask
    loss = jnp.sum(loss) / jnp.sum(label_mask)
    return loss


if __name__ == "__main__":
    train_and_eval()
