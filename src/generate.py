import jax.numpy as jnp
import orbax.checkpoint

from .model import GPT
from .model import GPTConfig

# TODO:
# - Include config in the checkpoint and use that.
# - Include tokenizer config in the checkpoint too?
# - Include key-value caching?


def generate():
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    state = checkpointer.restore("outputs/20230505_150044/checkpoints/857790/default")
    params = state["params"]

    config = GPTConfig()
    model = GPT(config)

    inputs = jnp.array([[0, 1, 23]])
    outputs = model.apply(params, inputs, train=False)
    breakpoint()

if __name__ == "__main__":
    generate()
