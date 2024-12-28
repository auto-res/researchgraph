from datasets import load_dataset
import re

dataset_name = "openai/gsm8k"

def custom_load_dataset():
    """
        download dataset from hugginface hub

    """

    return load_dataset(dataset_name, "main")

def parse_answer(answer):

    return int(match.group(1))\
    if (match := re.search(r"#### (\d+)", answer))\
    else None