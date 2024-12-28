import os
from huggingface_hub import login


hf_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
hf_user_name = os.getenv("HUGGINGFACE_USER_NAME")


def register_repository(dataset):
    login(token=hf_token)

    dataset.push_to_hub(f"{hf_user_name}/MATH", private=False)

