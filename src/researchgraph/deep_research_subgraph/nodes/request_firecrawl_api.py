import os
from pydantic import BaseModel
import httpx
import asyncio

FIRE_CRAWL_API_KEY = os.getenv("FIRE_CRAWL_API_KEY")


class SearchResponseItem(BaseModel):
    url: str
    markdown: str


class SearchResponse(BaseModel):
    search_data: list[SearchResponseItem]


async def request_firecrawl_api(
    query: str,
) -> SearchResponse:
    response = await _request_firecrawl(query)
    parsed_response = _parse_response(response)
    return parsed_response


async def _request_firecrawl(
    query: str, max_retries: int = 30, base_delay: float = 5, max_delay: float = 60
):
    # NOTE:The following is the official documentation of the API.
    # https://docs.firecrawl.dev/api-reference/endpoint/search
    url = "https://api.firecrawl.dev/v0/search"
    headers = {
        "Authorization": f"Bearer {FIRE_CRAWL_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "query": f"{query}",
        "timeout": 15000,
        "limit": 10,
        "scrapeOptions": {"formats": "markdown"},
    }
    delay = base_delay
    timeout = httpx.Timeout(60.0)
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=data, headers=headers)
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:  # Too Many Requests
                retry_after = e.response.headers.get("Retry-After")
                if retry_after:
                    delay = float(retry_after)
                print(f"Rate limited (429). Retrying in {delay:.2f} seconds...")

            elif e.response.status_code >= 500:  # サーバーエラー（5xx）
                print(
                    f"Server error {e.response.status_code}. Retrying in {delay:.2f} seconds..."
                )

            else:
                raise

        except httpx.RequestError as e:
            print(f"Request error: {e}. Retrying in {delay:.2f} seconds...")
        await asyncio.sleep(delay)
        delay = min(delay * 2, max_delay)
    raise Exception(f"Failed to fetch from Firecrawl API after {max_retries} retries.")


def _parse_response(response: dict) -> SearchResponse:
    search_items = [
        SearchResponseItem(url=item.get("url", ""), markdown=item.get("markdown"))
        for item in response.get("data", [])
    ]
    return search_items


async def main():
    query = "deepseekのアルゴリズムについて教えてください．"
    response = await request_firecrawl_api(query)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
