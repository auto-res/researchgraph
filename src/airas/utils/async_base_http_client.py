import httpx
import logging
from abc import ABC, abstractmethod
from researchgraph.utils.parser_mixin import ResponseParserMixIn

logger = logging.getLogger(__name__)


class AsyncBaseHTTPClient(ResponseParserMixIn, ABC):
    def __init__(
        self, 
        base_url: str, 
        default_headers: dict[str, str] | None = None
    ):
        self.base_url = base_url.rstrip("/")
        self.default_headers = default_headers or {}
        self.session = httpx.AsyncClient()

    async def _do_request():
        ...
        
    @abstractmethod
    async def _request_with_retry():
        raise NotImplementedError

    async def request():
        ...