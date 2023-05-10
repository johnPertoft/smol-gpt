from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint
from datasets import Dataset
from flax.training.train_state import TrainState
from flax import traverse_util
from tensorboardX import SummaryWriter

from .data import get_dataset_and_tokenizer
from .model import GPT
from .model import GPTConfig


# TODO:
# - Use a real dataloader of some sort?
# - Write this with multi gpu + multi host support? for fun
# - Add gradient accumulation.
# - Show a training summary before starting training.
# - Fix missing typing.
# - What weight decay makes sense?
# - Make restoring work well
#   - Include data state, or just skip ahead etc.
#   - Make epoch and steps make sense when restoring.
#   - Include train and model config too?
# - Save the config in the checkpoints too.
# - Use chex or tjax for pytree compatible dataclasses maybe?
#   Needed to be able to save configs in the checkpoints.


@dataclass
class TrainingConfig:
    seed: int = 123
    num_epochs: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    learning_rate_warmup_steps: int = 0
    weight_decay: float = 0.0
    gradient_clipping: float = 1.0


def train_and_eval(model: GPT, train_config: TrainingConfig, output_dir: Path):
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=3)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        checkpoint_dir,
        orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()),
        options=checkpoint_options,
    )
    
    dataset, tokenizer = get_dataset_and_tokenizer(256)

    rng = jax.random.PRNGKey(train_config.seed)
    params_rng, dropout_rng = jax.random.split(key=rng)
    rngs = {"params": params_rng, "dropout": dropout_rng}
    params = model.init(
        rngs=rngs,
        input_ids=jnp.empty((1, model.config.n_positions), dtype=jnp.int32),
        train=True,
    )

    num_epoch_steps = len(dataset["train"]) // train_config.per_device_batch_size
    num_train_steps = num_epoch_steps * train_config.num_epochs
    learning_rate_schedule = create_learning_rate_scheduler(
        learning_rate=train_config.learning_rate,
        warmup_steps=train_config.learning_rate_warmup_steps,
        total_train_steps=num_train_steps,
    )
    optimizer = create_optimizer(
        learning_rate=learning_rate_schedule,
        weight_decay=train_config.weight_decay,
        gradient_clipping=train_config.gradient_clipping,
    )
    # TODO: Need to think about step counter when using this.
    # E.g. is it taken care of for the lr schedule?
    # optimizer = optax.MultiSteps(optimizer, train_config.gradient_accumulation_steps)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params.unfreeze(),
        tx=optimizer,
    )
    
    # Restore if there's something to restore from.
    if checkpoint_manager.latest_step() is not None:
        # TODO: Restore data state etc.
        state = checkpoint_manager.restore(step=checkpoint_manager.latest_step(), items=state)
    
    summary_writer = SummaryWriter(output_dir / "logs")
    train_loss_ema = None
    
    for epoch in range(train_config.num_epochs):
        # Run train epoch.
        train_data_loader = create_data_loader(
            dataset["train"],
            batch_size=train_config.per_device_batch_size,
            rng=jax.random.fold_in(rng, epoch),
        )
        for batch in train_data_loader:
            lr = learning_rate_schedule(state.step)
            state, loss, grads = train_step(state, batch, jax.random.fold_in(rng, state.step))
            
            # Write tensorboard summaries.
            summary_writer.add_scalar("lr", lr, state.step)
            summary_writer.add_scalar("train/loss", loss, state.step)
            if state.step > 0 and state.step % 1000 == 0:
                jax.tree_util.tree_map_with_path(
                    lambda path, x: summary_writer.add_histogram(
                        "grads/" + "/".join(p.key for p in path), x, state.step
                    ),
                    grads["params"],
                )
                # TODO: Do this for params too?
            
            # Print progress info to console.
            if train_loss_ema is None:
                train_loss_ema = loss
            else:
                alpha = 0.1
                train_loss_ema = alpha * loss + (1 - alpha) * train_loss_ema
            progress_info = " - ".join([
                f"epoch {epoch + 1:02}",
                f"step {state.step:05} ({state.step / num_train_steps * 100:.2f}%)",
                f"lr: {lr:.3E}",
                f"loss: {loss:.3f} ({train_loss_ema:.3f})",
                f"ppl: {jnp.exp(loss):.3f}",
            ])
            print(progress_info)

        checkpoint_manager.save(state.step, state)

        # Run eval epoch.
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
        print("*" * 80)
        print(f"epoch {epoch + 1:02} - loss: {eval_loss:.3f} - ppl: {jnp.exp(eval_loss):.3f}")
        print("*" * 80)
        summary_writer.add_scalar("eval/loss", eval_loss, state.step)


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
    return state, loss, grads


@jax.jit
def eval_step(state: TrainState, batch: Dict[str, Any]) -> float:
    inputs = batch["input_ids"]
    labels = batch["labels"]
    logits = state.apply_fn(state.params, inputs, train=False)
    loss = loss_fn(logits, labels)
    return loss


def create_learning_rate_scheduler(learning_rate: float, warmup_steps: int, total_train_steps: int):
    warmup_lr_fn = optax.linear_schedule(
        init_value=0.0, end_value=learning_rate, transition_steps=warmup_steps
    )
    decay_lr_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0.0, transition_steps=total_train_steps - warmup_steps
    )
    return optax.join_schedules(schedules=[warmup_lr_fn, decay_lr_fn], boundaries=[warmup_steps])


def create_optimizer(learning_rate, weight_decay: float, gradient_clipping: float):
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

    return optax.chain(
        optax.clip(gradient_clipping),
        optax.adamw(learning_rate, weight_decay=weight_decay, mask=weight_decay_mask_fn)
    )


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


def loss_fn(logits, labels) -> float:
    label_mask = jax.numpy.where(labels > 0, 1.0, 0.0)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels) * label_mask
    loss = jnp.sum(loss) / jnp.sum(label_mask)
    return loss


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / timestamp
    output_dir.mkdir(parents=True)

    model_config = GPTConfig()
    model = GPT(model_config)
    config = TrainingConfig(
        num_epochs=15,
        per_device_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        learning_rate_warmup_steps=800,
        weight_decay=0.01,
    )
    train_and_eval(model, config, output_dir)
