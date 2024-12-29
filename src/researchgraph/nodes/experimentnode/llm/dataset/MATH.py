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

def parse_answer(answer):
    return _strip_string(
        remove_boxed(
            last_boxed_only_string(answer)
        )
    )


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


######################################
### function for parse answer
###   ref: https://github.com/hendrycks/math/blob/main/modeling/eval_math_gpt.py
######################################


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None
    




if __name__ == "__main__":
    # register_dataset()

    dataset_name = hf_dataset_name()
    print(f"dataset name: {dataset_name}")
    # dataset = load_dataset(dataset_name)

    # for i in dataset["train"]:
    #     sample_answer = i
    #     break

    sample_answer='The question is asking us to divide \\frac{1}{6}\\div \\frac{1}{3}$. To see this, imagine that the numbers were something nicer, for example: "How many threes are in 12?" We can see that this problem is asking us how many groups of 3 you can make if you have 12 things, and the answer is $12\\div 3=4$. So we get\\[\\frac{1}{6}\\div \\frac{1}{3} = \\frac{1}{6}\\cdot\\frac{3}{1}=\\frac{3}{6}=\\frac{1\\cdot\\cancel{3}}{2\\cdot \\cancel{3}}=\\boxed{\\frac{1}{2}}.\\]'

    print(f"=== sample === \n{sample_answer}")
    print(f"=== parsed sample === \n{parse_answer(sample_answer)}")

    sample_answer = "\\boxed{0.5}"

    print(f"=== sample === \n{sample_answer}")
    print(f"=== parsed sample === \n{parse_answer(sample_answer)}")

