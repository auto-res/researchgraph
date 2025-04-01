import tiktoken
from openai import OpenAI
from typing import Dict, List, Type
import json
from pydantic import BaseModel


# モデルごとの最大トークン数定義
MODEL_MAX_TOKENS = {
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
    return len(enc.encode(text))


def truncate_prompt(
    model_name: str, message: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """最大トークン数を超えないようにプロンプトを短縮"""
    max_tokens = MODEL_MAX_TOKENS.get(model_name, 4096)  # デフォルト4096
    enc = tiktoken.encoding_for_model(model_name)
    total_tokens = sum(len(enc.encode(msg["content"])) for msg in message)

    if total_tokens > max_tokens:
        print("警告: プロンプトが最大トークン数を超えています。短縮します。")

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
    model_name: str, message: List[Dict[str, str]], data_class: Type[BaseModel]
) -> dict:
    client = OpenAI()
    message = truncate_prompt(model_name, message)

    response = client.responses.create(
        model=model_name,
        input=message,
        response_format=data_class,
    )
    output = json.loads(response.choices[0].message.content)
    return output
