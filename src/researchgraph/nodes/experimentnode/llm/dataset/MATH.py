from datasets import load_dataset
import urllib.request
import tarfile
import os

from dataset_util import register_repository


hf_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
hf_user_name = os.getenv("HUGGINGFACE_USER_NAME", "vetpath") # 一時的に自分のユーザー名を使用、将来的にはautoresのものに変更
dataset_name = f"{hf_user_name}/MATH" 


def hf_dataset_name():
    """
        download dataset from hugginface hub

    """
    return dataset_name


def register_dataset():
    dataset = make_dataset_from_official_source()
    register_repository(dataset)


def make_dataset_from_official_source():
    """
        dump dataset to local
        and convert it to a format suitable for post-processing.

        tarファイルをダウンロードして解凍
        キーをquestion / answerに書き換える。
    """
    url = "https://people.eecs.berkeley.edu/~hendrycks/MATH.tar"

    tmp_dir = "./tmp"
    save_path = f"{tmp_dir}/MATH.tar"
    extract_path = f"{tmp_dir}/MATH/extracted"
    data_dir = f"{tmp_dir}/MATH/extracted/MATH"

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    urllib.request.urlretrieve(url, save_path)

    

    with tarfile.open(save_path, "r") as tar:
        tar.extractall(path=extract_path)

    data_files = {
        "train": [
            os.path.join(data_dir, "train", sub_dir, "*.json")
            for sub_dir in os.listdir(os.path.join(data_dir, "train"))
        ],
        "test": [
            os.path.join(data_dir, "test", sub_dir, "*.json")
            for sub_dir in os.listdir(os.path.join(data_dir, "test"))
        ],
    }

    # Hugging Face Datasets に読み込む
    dataset = load_dataset(
        "json", 
        data_files=data_files,
        cache_dir=f"{tmp_dir}/MATH/cache"
    )

    dataset = dataset.map(
        lambda example: {
            "question": example["problem"],
            "answer": example["solution"]
        },
        remove_columns=["problem", "solution", "level", "type"]
    )

    return dataset


if __name__ == "__main__":
    register_dataset()
