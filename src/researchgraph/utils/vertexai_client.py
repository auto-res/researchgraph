import os
import time
from typing import Literal, TypedDict, Type, Any
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


def base_model_to_typeddict(data_model: type[BaseModel]) -> Type[Any]:
    annotations = get_type_hints(model)
    return TypedDict(data_model.__name__ + "Dict", annotations)


def vertexai_client(
    model_name: VERTEXAI_MODEL,
    prompt: str,
    data_model: type[BaseModel] | None = None,
    max_retries: int = 3,
    delay: int = 1,
) -> str | None:
    data_model = base_model_to_typeddict(data_model)

    client = genai.Client(api_key=VERTEX_AI_API_KEY)

    while True:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json", response_schema=data_model
                ),
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
        return output
    else:
        logger.error("Empty response from Vertex AI API.")
        return None


if __name__ == "__main__":

    class UserModel(BaseModel):
        name: str
        age: int
        email: str

    model_name = "gemini-2.0-flash-001"
    prompt = "Hello, how are you?"
    response = vertexai_client(model_name, prompt)
    print(response)
