import os
import time
import ast
from typing import Literal
from pydantic import BaseModel
from google import genai
from logging import getLogger

logger = getLogger(__name__)

VERTEX_AI_API_KEY = os.getenv("VERTEX_AI_API_KEY")
client = genai.Client(api_key=VERTEX_AI_API_KEY)

VERTEXAI_MODEL = Literal[
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001",
]

VERTEXAI_MODEL_INFO = {
    "gemini-2.0-flash-001": {
        "max_input_tokens": 1048576,
        "input_token_cost": 0.0000001,
        "output_token_cost": 0.0000004,
    },
    "gemini-2.0-flash-lite-001": {
        "max_input_tokens": 1048576,
        "input_token_cost": 0.000000075,
        "output_token_cost": 0.0000003,
    },
}


def vertexai_client(
    model_name: VERTEXAI_MODEL,
    message: str,  # TODO: OpenAI APIのようにlist[dict[str, str]]形式にする
    data_model: type[BaseModel] | None = None,
    max_retries: int = 30,
    delay: int = 1,
) -> dict | None:
    client = genai.Client(api_key=VERTEX_AI_API_KEY)

    while True:
        try:
            if data_model is None:
                response = client.models.generate_content(
                    model=model_name,
                    contents=message,
                )
                output = response.text
            else:
                # data_model = basemodel_to_typeddict(data_model)
                response = client.models.generate_content(
                    model=model_name,
                    contents=message,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": list[data_model],
                    },
                )
                output = response.text
            break
        except Exception as e:
            logger.warning(f"Error calling Vertex AI API: {e}")
            if max_retries > 0:
                logger.info(f"Retrying... ({max_retries} retries left)")
                max_retries -= 1
                time.sleep(delay)
            else:
                logger.error("Max retries reached. Exiting.")
                raise
    if output:
        if "null" in output:
            output = output.replace("null", "None")
        output_list = ast.literal_eval(output)
        # NOTE:Since gemini can receive multiple data models, it returns multiple outputs in list format. Here, only one data model is given, so only the first element is retrieved.
        FIRST_INDEX_OF_LIST = 0
        return output_list[FIRST_INDEX_OF_LIST]
    else:
        logger.error("Empty response from Vertex AI API.")
        return None


if __name__ == "__main__":

    class UserModel(BaseModel):
        name: str
        age: int
        email: str

    model_name = "gemini-2.0-flash-001"
    message = """
以下の文章から，名前，年齢，メールアドレスを抽出してください。
「田中太郎さん（35歳）は、東京在住のソフトウェアエンジニアです。現在、新しいAI技術の研究に取り組んでおり、業界内でも注目を集めています。お問い合わせは、taro.tanaka@example.com までお願いします。」
"""
    response = vertexai_client(
        model_name=model_name,
        message=message,
        data_model=UserModel,
    )
    print(response)
