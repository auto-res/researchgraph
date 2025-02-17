from typing import TypedDict
import asyncio

from researchgraph.deep_research_subgraph.nodes.generate_queries import generate_queries
from researchgraph.deep_research_subgraph.nodes.request_firecrawl_api import (
    request_firecrawl_api,
)
from researchgraph.deep_research_subgraph.nodes.process_serp_result import (
    process_serp_result,
)

from researchgraph.deep_research_subgraph.nodes.generate_queries import QueryInfo


class ResearchResult(TypedDict):
    learnings: list[str]
    visited_urls: list[str]


async def recursive_search(
    query: str,
    breadth: int,
    depth: int,
    learnings: list[str] = None,
    visited_urls: list[str] = None,
) -> dict:
    if learnings is None:
        learnings = []
    if visited_urls is None:
        visited_urls = []
    serp_queries_list = await generate_queries(
        llm_name="gpt-4o-mini-2024-07-18",
        query=query,
        num_queries=breadth,
    )

    tasks = []
    for serp_query in serp_queries_list.queries_list:
        task = _process_query(
            serp_query=serp_query,
            depth=depth,
            breadth=breadth,
            current_learnings=learnings.copy(),
            current_visited_urls=visited_urls.copy(),
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_learnings = []
    all_visited_urls = []
    for result in results:
        if isinstance(result, Exception):
            print(f"Error occurred: {result}")
            continue
        if (
            isinstance(result, dict)
            and "learnings" in result
            and "visited_urls" in result
        ):
            all_learnings.extend(result["learnings"])
            all_visited_urls.extend(result["visited_urls"])

    return {
        "learnings": list(dict.fromkeys(all_learnings)),
        "visited_urls": list(dict.fromkeys(all_visited_urls)),
    }


async def _process_query(
    serp_query: QueryInfo,
    depth: int,
    breadth: int,
    current_learnings: list[str],
    current_visited_urls: list[str],
) -> ResearchResult:
    print(f"Processing query: {serp_query.query}")
    print(f"depth:{depth}")
    print(f"breadth:{breadth}")
    try:
        # 検索
        search_result = await request_firecrawl_api(serp_query.query)
        new_urls = [item.url for item in search_result if item.url]

        # 結果をまとめる
        processed_result = await process_serp_result(
            llm_name="gpt-4o-mini-2024-07-18",
            query=serp_query.query,
            result=search_result,
        )
        all_learnings = current_learnings + processed_result.learnings
        all_urls = current_visited_urls + new_urls

        if depth > 0:
            new_breadth = max(1, breadth // 2)
            new_depth = depth - 1
            next_query = (
                f"Previous research goal: {serp_query.research_goal}\n"
                f"Follow-up research directions:\n"
                f"{chr(10).join(processed_result.followup_questions)}"
            )
            return await recursive_search(
                query=next_query,
                breadth=new_breadth,
                depth=new_depth,
                learnings=all_learnings,
                visited_urls=all_urls,
            )

        return ResearchResult(learnings=all_learnings, visited_urls=all_urls)

    except Exception as e:
        print(f"Error processing query '{serp_query.query}': {str(e)}")
        return ResearchResult(learnings=[], visited_urls=[])


async def main():
    query = "Tell me more about the technology used in DeepSeek."
    response = await recursive_search(
        query=query,
        breadth=2,
        depth=1,
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
