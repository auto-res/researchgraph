import tiktoken
from openai import OpenAI
from typing import Optional, Dict
import json

# モデルごとの最大トークン数定義
MODEL_MAX_TOKENS = {
    "gpt-4.5-preview-2025-02-27": 16384 ,
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

def truncate_prompt(model_name: str, prompt: str) -> str:
    """最大トークン数を超えないようにプロンプトを短縮"""
    max_tokens = MODEL_MAX_TOKENS.get(model_name, 4096)  # デフォルト4096
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(prompt)
    if len(tokens) > max_tokens:
        print("警告: プロンプトが最大トークン数を超えています。短縮します。")
        tokens = tokens[:max_tokens]
    return enc.decode(tokens)

def openai_client(model_name: str, prompt: str, schema: Optional[Dict] = None) -> str:
    client = OpenAI()
    max_tokens = MODEL_MAX_TOKENS.get(model_name, 4096)  # デフォルト4096
    prompt = truncate_prompt(model_name, prompt)
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    text_format = (
        {
            "format": {
                "type": "json_schema",
                "name": "structured_response",
                "schema": schema,
                "strict": True,
            }
        }
        if schema else None
    )
    
    response = client.responses.create(
        model=model_name,
        input=messages,
        text=text_format
    )
    
    output = json.loads(response.output_text) if schema else response.output_text
    return output