import tiktoken
from openai import OpenAI
from pydantic import BaseModel
import time
from typing import Literal
from logging import getLogger

logger = getLogger(__name__)


OPENAI_MODEL = Literal[
    "o3-mini-2025-01-31",
    "o1-2024-12-17",
    "o1-mini-2024-09-12",
    "gpt-4.5-preview-2025-02-27",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-11-20",
]


# Maximum number of tokens per model definition
MODEL_MAX_TOKENS = {
    "o3-mini-2025-01-31": 200000,  # context window(100000 max output tokens)
    "o1-2024-12-17": 128000,  # context window(32768 max output tokens)
    "o1-mini-2024-09-12": 128000,  # context window(65536 max output tokens)
    "gpt-4.5-preview-2025-02-27": 16384,
    "gpt-4o-mini-2024-07-18": 16384,
    "gpt-4o-2024-11-20": 16384,
    "gpt-4-0613": 8192,
    "gpt-4-32k-0613": 32768,
    "gpt-4-1106-preview": 128000,
    "gpt-3.5-turbo-1106": 4096,
    "gpt-3.5-turbo-16k-0613": 16384,
    "gpt-3.5-turbo-0613": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
}


def count_tokens(model_name: str, text: str) -> int:
    """モデルに応じたトークン数を計算"""
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text, disallowed_special=()))


def truncate_prompt(
    model_name: OPENAI_MODEL, message: list[dict[str, str]]
) -> list[dict[str, str]]:
    """最大トークン数を超えないようにプロンプトを短縮"""
    max_tokens = MODEL_MAX_TOKENS.get(model_name, 4096)  # デフォルト4096
    enc = tiktoken.encoding_for_model(model_name)
    total_tokens = sum(len(enc.encode(msg["content"])) for msg in message)

    if total_tokens > max_tokens:
        logger.warning(f"Prompt length exceeds {max_tokens} tokens. Truncating.")

        # content のトークン数が多いものから順に削る
        for msg in message[::-1]:
            if total_tokens <= max_tokens:
                break
            tokens = enc.encode(msg["content"])
            if len(tokens) > max_tokens // len(message):
                msg["content"] = enc.decode(tokens[: max_tokens // len(message)])
            total_tokens = sum(len(enc.encode(m["content"])) for m in message)

    return message


def openai_client(
    model_name: str,
    message: list[dict[str, str]],
    data_model: type[BaseModel] | None = None,
    max_retries: int = 30,
    delay: int = 1,
) -> str | None:
    client = OpenAI()

    # NOTE：エンコーディング，デコーディングの処理とtiktokenの処理が冗長なので，後で修正する
    for msg in message:
        if "content" in msg:
            msg["content"] = msg["content"].encode("utf-8", "ignore").decode("utf-8")

    message = truncate_prompt(model_name, message)

    while True:
        try:
            if data_model is None:
                response = client.responses.create(
                    model=model_name,
                    input=message,
                )
                output = response.output_text
            else:
                response = client.beta.chat.completions.parse(
                    model=model_name,
                    messages=message,
                    response_format=data_model,
                )
                output = response.choices[0].message.content
            break
        except Exception as e:
            logger.warning(f"Error calling OpenAI API: {e}")
            if max_retries > 0:
                logger.info(f"Retrying... ({max_retries} retries left)")
                max_retries -= 1
                time.sleep(delay)
            else:
                logger.error("Max retries reached. Exiting.")
                raise
    if output:
        return output
    else:
        logger.error("Empty response from OpenAI API.")
        return None
