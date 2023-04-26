from dataclasses import dataclass

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import jax
import jax.numpy as jnp
import optax
from datasets import Dataset
from flax.training.train_state import TrainState
from flax import traverse_util

from .data import get_dataset_and_tokenizer
from .model import GPT
from .model import GPTConfig


# TODO:
# - Write this with more low level contructs? I.e. without TrainState.
# - Use a real dataloader of some sort?
# - Write this with multi gpu + multi host support? for fun
# - Save checkpoint.
# - Plot train and eval loss.

@dataclass
class TrainingConfig:
    seed: int = 123
    num_epochs: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    learning_rate_warmup_steps: int = 0


def train_and_eval(train_config: TrainingConfig):
    rng = jax.random.PRNGKey(train_config.seed)
    
    dataset, tokenizer = get_dataset_and_tokenizer(256)

    model_config = GPTConfig()
    model = GPT(model_config)
    params_rng, dropout_rng = jax.random.split(key=rng)
    rngs = {"params": params_rng, "dropout": dropout_rng}
    params = model.init(
        rngs=rngs,
        input_ids=jnp.empty((1, model_config.n_positions), dtype=jnp.int32),
        train=True,
    )

    num_train_steps = len(dataset["train"]) // train_config.per_device_batch_size 
    optimizer = create_optimizer(
        learning_rate=train_config.learning_rate,
        warmup_steps=train_config.learning_rate_warmup_steps,
        total_train_steps=num_train_steps,
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    train_losses = []
    eval_losses = []
    for epoch in range(train_config.num_epochs):
        train_data_loader = create_data_loader(
            dataset["train"],
            batch_size=train_config.per_device_batch_size,
            rng=jax.random.fold_in(rng, epoch),
        )
        for batch in train_data_loader:
            state, loss = train_step(state, batch, jax.random.fold_in(rng, state.step))
            print(f"Epoch {epoch + 1:02} Step {state.step:04} - Loss: {loss:.3f} - Ppl: {jnp.exp(loss):.3f}")
            train_losses.append(loss)

        total_eval_loss = 0.0
        eval_loss_samples = 0
        eval_data_loader = create_data_loader(
            dataset["validation"],
            batch_size=train_config.per_device_batch_size,
        )
        for batch in eval_data_loader:
            loss = eval_step(state, batch)
            total_eval_loss += loss
            eval_loss_samples += 1
        eval_loss = total_eval_loss / eval_loss_samples
        print("=" * 80)
        print(f"Epoch {epoch + 1:02} - Loss: {eval_loss:.3f} - Ppl: {jnp.exp(eval_loss):.3f}")
        print("=" * 80)
        eval_losses.append((state.step, eval_loss))

    import matplotlib.pyplot as plt
    plt.plot(train_losses, label="train")
    plt.plot([x[0] for x in eval_losses], [x[1] for x in eval_losses], label="eval")
    plt.show()


def create_optimizer(learning_rate: float, warmup_steps: int, total_train_steps: int):
    warmup_lr_fn = optax.linear_schedule(
        init_value=0.0, end_value=learning_rate, transition_steps=warmup_steps
    )
    decay_lr_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0.0, transition_steps=total_train_steps - warmup_steps
    )
    lr_fn = optax.join_schedules(schedules=[warmup_lr_fn, decay_lr_fn], boundaries=[warmup_steps])

    def weight_decay_mask_fn(params):
        def is_masked_layer_norm_key(k):
            # TODO: This is kind of ugly. Because it depends on the naming we picked
            # for the layer norm layers.
            layer_name = k[-2]
            param_name = k[-1]
            is_layer_norm = layer_name.startswith("ln") or layer_name.startswith("layer_norm")
            if is_layer_norm and param_name == "scale":
                return True
            return False
        
        # Create a PyTree with the same structure as params, but with boolean leaf nodes.
        # True for params that should be decayed and False for params that should not be decayed.
        flattened_params = traverse_util.flatten_dict(params)
        flattened_mask_tree = {k: not is_masked_layer_norm_key(k) for k in flattened_params.keys()}
        return traverse_util.unflatten_dict(flattened_mask_tree)
    
    return optax.adamw(lr_fn, mask=weight_decay_mask_fn)

"""
def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        # find out all LayerNorm parameters
        layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
        layer_norm_named_params = {
            layer[-2:]
            for layer_norm_name in layer_norm_candidates
            for layer in flat_params.keys()
            if layer_norm_name in "".join(layer).lower()
        }
        flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)
"""

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


@jax.jit
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
    

@jax.jit
def eval_step(state: TrainState, batch: Dict[str, Any]) -> float:
    inputs = batch["input_ids"]
    labels = batch["labels"]
    logits = state.apply_fn(state.params, inputs, train=False)
    loss = loss_fn(logits, labels)
    return loss


def loss_fn(logits, labels) -> float:
    label_mask = jax.numpy.where(labels > 0, 1.0, 0.0)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels) * label_mask
    loss = jnp.sum(loss) / jnp.sum(label_mask)
    return loss


if __name__ == "__main__":
    config = TrainingConfig(
        num_epochs=5,
        per_device_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        learning_rate_warmup_steps=5000,
    )
    train_and_eval(config)
