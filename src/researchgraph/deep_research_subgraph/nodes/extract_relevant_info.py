from dataclasses import dataclass
from typing import Optional, TypeVar
import os
import tiktoken
from openai import AsyncOpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class AIProviderConfig:
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-3.5-turbo"


class OpenAIProvider:
    def __init__(self, config: Optional[AIProviderConfig] = None):
        """
        Initialize OpenAI provider with optional custom configuration.
        If no config provided, reads from environment variables.
        """
        if config is None:
            config = AIProviderConfig(
                api_key=os.getenv("OPENAI_KEY", ""),
                base_url=os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1"),
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            )

        self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        self.model = config.model
        self._encoder = tiktoken.get_encoding("cl100k_base")

    async def generate_object(
        self, system: str, prompt: str, schema: type[T], max_tokens: int = 4000
    ) -> T:
        """
        Generate a structured response using the OpenAI API.

        Args:
            system: System prompt
            prompt: User prompt
            schema: Pydantic model class for response validation
            max_tokens: Maximum tokens for response

        Returns:
            Validated response object
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        # Parse and validate response using the provided schema
        content = response.choices[0].message.content
        return schema.model_validate_json(content)

    def trim_prompt(
        self, prompt: str, context_size: int = 128_000, min_chunk_size: int = 140
    ) -> str:
        """
        Trim prompt to fit within context size while preserving meaning.

        Args:
            prompt: Input text
            context_size: Maximum context size in tokens
            min_chunk_size: Minimum chunk size to preserve

        Returns:
            Trimmed prompt text
        """
        if not prompt:
            return ""

        tokens = self._encoder.encode(prompt)
        if len(tokens) <= context_size:
            return prompt

        # Calculate approximate character length based on token overflow
        overflow_tokens = len(tokens) - context_size
        chunk_size = len(prompt) - (
            overflow_tokens * 3
        )  # Approximate 3 chars per token

        if chunk_size < min_chunk_size:
            return prompt[:min_chunk_size]

        # Recursively trim until within context size
        return self.trim_prompt(prompt[:chunk_size], context_size)


# System prompt template
def get_system_prompt() -> str:
    """Generate system prompt with current timestamp."""
    from datetime import datetime

    now = datetime.utcnow().isoformat()

    return f"""You are an expert researcher. Today is {now}. Follow these instructions when responding:
    - You may be asked to research subjects that is after your knowledge cutoff, assume the user is right when presented with news.
    - The user is a highly experienced analyst, no need to simplify it, be as detailed as possible and make sure your response is correct.
    - Be highly organized.
    - Suggest solutions that I didn't think about.
    - Be proactive and anticipate my needs.
    - Treat me as an expert in all subject matter.
    - Mistakes erode my trust, so be accurate and thorough.
    - Provide detailed explanations, I'm comfortable with lots of detail.
    - Value good arguments over authorities, the source is irrelevant.
    - Consider new technologies and contrarian ideas, not just the conventional wisdom.
    - You may use high levels of speculation or prediction, just flag it for me."""
