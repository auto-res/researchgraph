import requests
import logging
from abc import ABC, abstractmethod
from researchgraph.utils.parser_mixin import ResponseParserMixIn

logger = logging.getLogger(__name__)


class BaseHTTPClient(ResponseParserMixIn, ABC):
    def __init__(
        self,
        base_url: str,
        default_headers: dict[str, str] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_headers = default_headers or {}
        self.session = requests.Session()

    def _do_request(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict | None = None,
        json: dict | None = None,
        stream: bool = False,
        timeout: float = 10.0,
    ) -> requests.Response:
        url = f"{self.base_url}/{path.lstrip('/')}"
        hdrs = {**self.default_headers, **(headers or {})}
        resp = self.session.request(
            method=method,
            url=url,
            headers=hdrs,
            params=params,
            json=json,
            timeout=timeout,
            stream=stream,
        )
        resp.raise_for_status()
        return resp


    @abstractmethod
    def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict | None = None,
        json: dict | None = None,
        stream: bool = False,
        timeout: float = 10.0,
    ) -> requests.Response:
        """
        Abstract method to be implemented in subclasses with a retry decorator.

        Example:
            @retry(...)
            def _request_with_retry(...):
                return super()._send(...)
        """
        raise NotImplementedError

    def request(
        self, 
        method: str, 
        path: str, 
        *, 
        headers: dict[str, str] | None = None,
        params: dict | None = None,
        json: dict | None = None,
        stream: bool = False,
        timeout: float = 10.0,
    ) -> dict | str | bytes | None:
        try:
            resp = self._request_with_retry(
                method.upper(),
                path,
                headers=headers,
                params=params,
                json=json,
                stream=stream,
                timeout=timeout,
            )
            return self._parse_response(resp)
        except Exception as e:
            logger.warning(f"Error during API request: {e}")
            return None # TODO