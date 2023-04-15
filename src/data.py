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
    dataset = load_dataset("tiny_shakespeare")

    def chunk_input_ids(x):
        assert len(x["input_ids"]) == 1, "Batch size must be 1."
        input_ids = x["input_ids"][0]
        input_ids_chunks = [input_ids[i:i + chunk_size] for i in range(0, len(input_ids), chunk_size)]
        return {"input_ids": input_ids_chunks}
    
    dataset = dataset.map(
        lambda x: {"input_ids": tokenizer.encode(x["text"]).ids},
        remove_columns=["text"],
    )
    dataset = dataset.map(chunk_input_ids, batch_size=1, batched=True)
    dataset = dataset.filter(lambda x: len(x["input_ids"]) == chunk_size)
    dataset = dataset.map(lambda x: {"input_ids": x["input_ids"][:-1], "labels": x["input_ids"][1:]})

    return dataset, tokenizer
