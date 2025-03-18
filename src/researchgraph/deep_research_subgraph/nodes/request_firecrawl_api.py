import os
from pydantic import BaseModel
import asyncio
from typing import List

from researchgraph.utils.firecrawl_app import FirecrawlApp

FIRE_CRAWL_API_KEY = os.getenv("FIRE_CRAWL_API_KEY")


class SearchResponseItem(BaseModel):
    url: str
    markdown: str = ""
    title: str = ""


async def request_firecrawl_api(
    query: str,
) -> List[SearchResponseItem]:
    """
    FireCrawl APIを使用してウェブ検索を実行する

    Args:
        query: 検索クエリ

    Returns:
        検索結果のリスト
    """
    print(f"  Sending FireCrawl API request for query: '{query}'")

    try:
        # 新しいFirecrawlAppクラスを使用
        firecrawl = FirecrawlApp(FIRE_CRAWL_API_KEY)

        # APIキーのデバッグ
        if not FIRE_CRAWL_API_KEY:
            print(f"  WARNING: FIRE_CRAWL_API_KEY environment variable is not set")
        else:
            masked_key = FIRE_CRAWL_API_KEY[:6] + "..." + FIRE_CRAWL_API_KEY[-4:] if len(FIRE_CRAWL_API_KEY) > 10 else "***"

        # 検索オプションの設定
        scrape_options = {"formats": ["markdown"]}

        # リクエストの詳細をログ出力
        print(f"  Request URL: {firecrawl.api_url}/search")
        print(f"  Request options: query='{query}', limit=10, scrapeOptions={scrape_options}")

        # 検索実行
        print(f"  Executing FireCrawl API search...")
        response = await firecrawl.search(
            query=query,
            timeout=15000,  # 15秒タイムアウト
            limit=10,       # 最大10件の結果
            scrape_options=scrape_options
        )

        # レスポンスの詳細をログ出力
        print(f"  Response structure: {list(response.keys())}")

        # 検索結果のパース
        parsed_response = _parse_response(response)
        print(f"  Received {len(parsed_response)} results from FireCrawl API")
        return parsed_response

    except Exception as e:
        print(f"  Error with FireCrawl API: {e}")
        print(f"  FireCrawl API failed - returning empty results")
        # APIが失敗した場合は空のリストを返す
        return []


def _parse_response(response: dict) -> List[SearchResponseItem]:
    """
    FireCrawl APIのレスポンスをパースする

    Args:
        response: APIレスポンス（JSON）

    Returns:
        検索結果のリスト
    """
    # データフィールドの取得
    data = response.get("data", [])
    if not data:
        print(f"  Warning: No data in response - returning empty results")
        return []

    # 検索結果の変換
    search_items = []
    for item in data:
        if not item:
            continue

        url = item.get("url", "")
        if not url:
            continue

        # マークダウンコンテンツの取得（なければテキストを使用）
        markdown = item.get("markdown", "")
        if not markdown:
            markdown = item.get("text", "")

        title = item.get("title", "")

        search_items.append(
            SearchResponseItem(
                url=url,
                markdown=markdown,
                title=title
            )
        )

    return search_items


async def main():
    """テスト用のメイン関数"""
    # 環境変数からAPIキーを取得
    api_key = os.getenv("FIRE_CRAWL_API_KEY")
    if not api_key:
        print("Error: FIRE_CRAWL_API_KEY environment variable not set")
        return

    query = "latest advancements in deep learning"
    print(f"Testing FireCrawl API with query: '{query}'")

    try:
        response = await request_firecrawl_api(query)
        print(f"Found {len(response)} results:")

        for i, item in enumerate(response, 1):
            print(f"\nResult {i}:")
            print(f"Title: {item.title}")
            print(f"URL: {item.url}")

            # マークダウンの一部を表示
            content_preview = item.markdown[:200] + "..." if len(item.markdown) > 200 else item.markdown
            print(f"Content preview: {content_preview}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
