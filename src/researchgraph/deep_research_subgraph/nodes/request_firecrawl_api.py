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
):
    response = await _request_firecrawl(query)
    parsed_response = _parse_response(response)
    return parsed_response


async def _request_firecrawl(
    query: str,
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
        "limit": 5,
        "scrapeOptions": {"formats": "markdown"},
    }
    timeout = httpx.Timeout(60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()


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
