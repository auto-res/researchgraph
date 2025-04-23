import requests
from abc import ABC, abstractmethod
from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential,
    retry_if_exception_type, 
    # retry_if_result, 
    before_log, 
    before_sleep_log, 
)
import logging

logger = logging.getLogger(__name__)

# Retry on exceptions or if response is None
retry_on = (
    retry_if_exception_type((HTTPError, ConnectionError, Timeout, RequestException, Exception))
    # | retry_if_result(lambda resp: resp is None)
)


class BaseAPIClient(ABC):
    def __init__(
        self,
        base_url: str,
        default_headers: dict[str, str] | None = None,
        max_retries: int = 5,
        initial_wait: float = 1.0,
        max_wait: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.default_headers = default_headers or {}
        self.max_retries = max_retries
        self.initial_wait = initial_wait
        self.max_wait = max_wait

    @retry(
        retry=retry_on, 
        stop=stop_after_attempt(20),
        wait=wait_exponential(multiplier=1.0, max=180.0), # TODO: インスタンス引数の反映
        before=before_log(logger, logging.WARNING),
        before_sleep=before_sleep_log(logger, logging.WARNING), 
        reraise=True
    )
    def _request(
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

    def _parse_response(self, resp: requests.Response) -> dict | str | bytes | None:
        content_type = resp.headers.get("Content-Type", "").lower()
        # JSON response
        if "application/json" in content_type:
            return resp.json() if resp.text.strip() else {}
        
        # No Content
        if resp.status_code == 204:
            return None
        
        # Binary (ZIP or octet-stream)
        if any(bin_ct in content_type for bin_ct in[
            "application/zip",
            "application/octet-stream",
            "application/x-zip-compressed",
        ]):
            if resp.raw:
                return resp
            return resp.content
        
        # XML/Atom
        if "xml" in content_type:
            return resp
        
        # Text response
        if "text/" in content_type:
            return resp.text.strip()
        
        # Unknown format
        logger.warning(f"Unknown Content-Type '{content_type}', returning raw bytes.")
        return resp.content

    @abstractmethod
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
            resp = self._request(
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
            return None
