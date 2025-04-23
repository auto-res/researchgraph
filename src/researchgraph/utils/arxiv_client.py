from typing import Any
from logging import getLogger
from researchgraph.utils.base_api_client import BaseAPIClient

logger = getLogger(__name__)


class ArxivClient(BaseAPIClient):
    def __init__(
        self,
        base_url: str = "https://export.arxiv.org/api",
        default_headers: dict[str, str] | None = None,
        max_retries: int = 20,
        initial_wait: float = 1.0,
        max_wait: float = 180.0,
    ):
        super().__init__(
            base_url=base_url,
            default_headers=default_headers,
            max_retries=max_retries,
            initial_wait=initial_wait,
            max_wait=max_wait,
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: dict | None = None,
        stream: bool = False,
        timeout: float = 10.0,
    ) -> dict | str | bytes | None:
        return super().request(
            method=method,
            path=path,
            headers=headers,
            params=params,
            json=json,
            stream=stream,
            timeout=timeout,
        )
