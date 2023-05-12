import jax.numpy as jnp
import orbax.checkpoint
from tokenizers import Tokenizer

from .model import GPT
from .model import GPTConfig


def generate():
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    state = checkpointer.restore("outputs/20230505_150044/checkpoints/857790/default")
    params = state["params"]

    config = GPTConfig()
    model = GPT(config)
    tokenizer = Tokenizer.from_pretrained("distilgpt2")

    prompt = "The Hustler is a 1961 American drama film"
    inputs = tokenizer.encode(prompt).ids
    for _ in range(64):
        outputs = model.apply(params, jnp.array([inputs]), train=False)
        next_token = outputs[0, -1, :].argmax()
        inputs.append(next_token)
    
    text = tokenizer.decode(inputs)
    breakpoint()

if __name__ == "__main__":
    generate()
