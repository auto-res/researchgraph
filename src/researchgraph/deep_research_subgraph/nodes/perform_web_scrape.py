import os
import asyncio
import urllib.parse
from pydantic import BaseModel
from researchgraph.utils.firecrawl_app import FirecrawlApp

FIRE_CRAWL_API_KEY = os.getenv("FIRE_CRAWL_API_KEY")


class SearchResponseItem(BaseModel):
    url: str
    markdown: str = ""
    title: str = ""

def _embed_query_in_url(url: str, query: str) -> str:

    return f"{url}&search={urllib.parse.quote_plus(query)}"

async def perform_web_scrape(
    query: str,
    scrape_urls: list
) -> list[SearchResponseItem]:
    
    if not FIRE_CRAWL_API_KEY:
            print("WARNING: FIRE_CRAWL_API_KEY environment variable is not set")
            return []
    
    fire_crawl = FirecrawlApp(FIRE_CRAWL_API_KEY)
    print("Executing FireCrawl API scraping...")

    scrape_params = {
        "formats": ["markdown"], 
        "onlyMainContent": True, 
        "waitFor": 5000, 
        "timeout": 15000, 
    }
    all_search_items = []
    try:
        for url in scrape_urls:
            full_url = _embed_query_in_url(url, query)
            print(f"Scraping URL: {full_url}")
            response = await fire_crawl.scrape_url(url=full_url, params=scrape_params)
            print(f"Response structure for {full_url}: {list(response.keys())}")
            parsed_response = _parse_response(response)
            print(f"Received {len(parsed_response)} results from FireCrawl API for URL: {full_url}")
            all_search_items.extend(parsed_response)
        return all_search_items

    except Exception as e:
        print(f"Error with FireCrawl API: {e}")
        print("FireCrawl API failed - returning empty results")
        return []


def _parse_response(response: dict) -> list[SearchResponseItem]:
    data = response.get("data", [])
    if not data:
        print(f"  Warning: No data in response - returning empty results")
        return []

    search_items = []
    if isinstance(data, dict):
        markdown = data.get("markdown", "")
        search_items.append(SearchResponseItem(url="", markdown=markdown, title=""))
    else:
        print("Warning: Unexpected data format - returning empty results")

    return search_items


async def main():
    api_key = os.getenv("FIRE_CRAWL_API_KEY")
    if not api_key:
        print("Error: FIRE_CRAWL_API_KEY environment variable not set")
        return

    query = "deep learning"
    scrape_urls = ["https://iclr.cc/virtual/2024/papers.html?filter=titles"]

    try:
        results = await perform_web_scrape(query, scrape_urls=scrape_urls)
        print(f"Found {len(results)} results:")

        for i, item in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Title: {item.title}")
            print(f"URL: {item.url}")

            content_preview = item.markdown[:200] + "..." if len(item.markdown) > 200 else item.markdown
            print(f"Content preview: {content_preview}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
