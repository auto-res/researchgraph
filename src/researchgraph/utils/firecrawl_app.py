import os
import httpx
import asyncio
from typing import Dict, Any, Optional, List

class FirecrawlApp:
    """Python implementation of FirecrawlApp similar to the TypeScript version."""

    def __init__(
        self, 
        api_key: Optional[str] = None, 
        api_url: Optional[str] = None, 
        max_retries: int = 20, 
        initial_wait_time = 1, 
        max_wait_time = 180, 
    ):
        """Initialize the FirecrawlApp with API key and optional base URL."""
        self.api_key = api_key or os.getenv("FIRE_CRAWL_API_KEY")  # Note: Using FIRE_CRAWL_API_KEY to match existing env var
        if not self.api_key:
            raise ValueError("FirecrawlApp requires an API key")

        self.api_url = api_url or "https://api.firecrawl.dev/v1"
        self.max_retries = max_retries
        self.initial_wait_time = initial_wait_time
        self.max_wait_time = max_wait_time

    async def search(self, query: str, timeout: int = 15000, limit: int = 5,
                    scrape_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search the web using Firecrawl API.

        Args:
            query: The search query
            timeout: Timeout in milliseconds
            limit: Maximum number of results
            scrape_options: Options for scraping

        Returns:
            Search response as a dictionary
        """
        url = f"{self.api_url}/search"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "query": query,
            "timeout": timeout,
            "limit": limit,
            "scrapeOptions": scrape_options or {"formats": ["markdown"]},
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            response = await client.post(url, json=data, headers=headers)
            response.raise_for_status()
            return response.json()

    async def map_url(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Map a website URL using Firecrawl API.

        Args:
            url: The website URL to map
            params: Additional parameters

        Returns:
            Mapping response as a dictionary
        """
        endpoint = f"{self.api_url}/map"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "url": url,
            **(params or {})
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            response = await client.post(endpoint, json=data, headers=headers)
            response.raise_for_status()
            return response.json()

    async def scrape_url(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Scrape a specific URL using Firecrawl API.

        Args:
            url: The URL to scrape
            params: Additional parameters

        Returns:
            Scraping response as a dictionary
        """
        endpoint = f"{self.api_url}/scrape"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "url": url,
            **(params or {})
        }

        retry_count = 0
        wait_time = self.initial_wait_time
        while retry_count < self.max_retries:
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                    response = await client.post(endpoint, json=data, headers=headers)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                print(f"Attempt {retry_count+1} failed with HTTP error: {e}")
            except Exception as e:
                print(f"Unexpected error on attempt {retry_count+1}: {e}")

            print(f"Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            wait_time = min(wait_time * 2, self.max_wait_time)
            retry_count += 1
        raise Exception("Max retries reached. Scrape URL API request failed.")

async def main():
    """Test the FirecrawlApp."""
    api_key = os.getenv("FIRE_CRAWL_API_KEY")
    if not api_key:
        print("Error: FIRE_CRAWL_API_KEY environment variable not set")
        return

    app = FirecrawlApp(api_key)
    query = "latest advancements in deep learning"
    print(f"Testing FirecrawlApp with query: '{query}'")

    try:
        response = await app.search(query)
        print(f"Response structure: {list(response.keys())}")

        data = response.get("data", [])
        print(f"Found {len(data)} results:")

        for i, item in enumerate(data, 1):
            print(f"\nResult {i}:")
            print(f"Title: {item.get('title', 'No title')}")
            print(f"URL: {item.get('url', 'No URL')}")

            # マークダウンの一部を表示
            markdown = item.get("markdown", "")
            content_preview = markdown[:200] + "..." if len(markdown) > 200 else markdown
            print(f"Content preview: {content_preview}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
