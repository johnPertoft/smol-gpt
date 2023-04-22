from itertools import chain

from datasets import load_dataset
from tokenizers import Tokenizer


def get_dataset_and_tokenizer(sequence_length: int):
    # TODO:
    # - Use another/bigger dataset.
    # - Maybe train the tokenizer on the dataset too.
    # - Use padding instead of skipping last.

    # Add one to sequence length to account for the right shifted labels.
    chunk_size = sequence_length + 1
    
    tokenizer = Tokenizer.from_pretrained("distilgpt2")
    dataset = load_dataset("wikitext", "wikitext-103-v1")

    def group_and_chunk_examples(examples):
        input_ids = list(chain(*examples["input_ids"]))
        input_ids_chunks = [input_ids[i:i + chunk_size] for i in range(0, len(input_ids), chunk_size)]
        if len(input_ids_chunks[-1]) < chunk_size:
            input_ids_chunks = input_ids_chunks[:-1]
        return {"input_ids": input_ids_chunks}

    dataset = dataset.map(
        lambda x: {"input_ids": tokenizer.encode(x["text"]).ids},
        remove_columns=["text"],
    )

    dataset = dataset.map(group_and_chunk_examples, batched=True)
    dataset = dataset.map(lambda x: {"input_ids": x["input_ids"][:-1], "labels": x["input_ids"][1:]})

    return dataset, tokenizer
