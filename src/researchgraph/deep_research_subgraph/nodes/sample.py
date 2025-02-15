from dataclasses import dataclass
from typing import List, Optional, Callable
import asyncio
from concurrent.futures import ThreadPoolExecutor


@dataclass
class ResearchProgress:
    current_depth: int
    total_depth: int
    current_breadth: int
    total_breadth: int
    current_query: Optional[str]
    total_queries: int
    completed_queries: int


@dataclass
class ResearchResult:
    learnings: List[str]
    visited_urls: List[str]


@dataclass
class SerpQuery:
    query: str
    research_goal: str


class DeepResearch:
    def __init__(
        self,
        concurrency_limit: int = 2,
        on_progress: Optional[Callable[[ResearchProgress], None]] = None,
    ):
        self.concurrency_limit = concurrency_limit
        self.on_progress = on_progress
        self.executor = ThreadPoolExecutor(max_workers=concurrency_limit)

    async def deep_research(
        self,
        query: str,
        breadth: int,
        depth: int,
        learnings: List[str] = None,
        visited_urls: List[str] = None,
    ) -> ResearchResult:
        """
        Main research function that orchestrates the deep research process.
        """
        if learnings is None:
            learnings = []
        if visited_urls is None:
            visited_urls = []

        progress = ResearchProgress(
            current_depth=depth,
            total_depth=depth,
            current_breadth=breadth,
            total_breadth=breadth,
            current_query=None,
            total_queries=0,
            completed_queries=0,
        )

        def report_progress(**kwargs):
            for key, value in kwargs.items():
                setattr(progress, key, value)
            if self.on_progress:
                self.on_progress(progress)

        # Generate SERP queries using AI
        serp_queries = await self._generate_serp_queries(query, breadth, learnings)
        report_progress(
            total_queries=len(serp_queries),
            current_query=serp_queries[0].query if serp_queries else None,
        )

        # Process queries concurrently
        tasks = []
        for serp_query in serp_queries:
            task = self._process_query(
                serp_query=serp_query,
                depth=depth,
                breadth=breadth,
                current_learnings=learnings.copy(),
                current_visited_urls=visited_urls.copy(),
                progress=progress,
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results, filtering out any errors
        all_learnings = []
        all_visited_urls = []
        for result in results:
            if isinstance(result, ResearchResult):
                all_learnings.extend(result.learnings)
                all_visited_urls.extend(result.visited_urls)

        # Remove duplicates while preserving order
        return ResearchResult(
            learnings=list(dict.fromkeys(all_learnings)),
            visited_urls=list(dict.fromkeys(all_visited_urls)),
        )

    async def _process_query(
        self,
        serp_query: SerpQuery,
        depth: int,
        breadth: int,
        current_learnings: List[str],
        current_visited_urls: List[str],
        progress: ResearchProgress,
    ) -> ResearchResult:
        """
        Process a single SERP query and recursively explore if depth allows.
        """
        try:
            # Search and extract content
            search_result = await self._search(serp_query.query)
            new_urls = [item.url for item in search_result.data if item.url]

            # Process results to extract learnings
            processed_result = await self._process_serp_result(
                query=serp_query.query, result=search_result
            )

            all_learnings = current_learnings + processed_result.learnings
            all_urls = current_visited_urls + new_urls

            # If we have more depth, continue research
            if depth > 0:
                new_breadth = max(1, breadth // 2)
                new_depth = depth - 1

                next_query = (
                    f"Previous research goal: {serp_query.research_goal}\n"
                    f"Follow-up research directions:\n"
                    f"{chr(10).join(processed_result.follow_up_questions)}"
                )

                return await self.deep_research(
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

    async def _generate_serp_queries(
        self, query: str, num_queries: int, learnings: List[str]
    ) -> List[SerpQuery]:
        """
        Generate SERP queries using AI. To be implemented by AI integration.
        """
        # This will be implemented in the AI integration component
        pass

    async def _search(self, query: str):
        """
        Perform web search. To be implemented by search integration.
        """
        # This will be implemented in the search integration component
        pass

    async def _process_serp_result(self, query: str, result):
        """
        Process search results using AI. To be implemented by AI integration.
        """
        # This will be implemented in the AI integration component
        pass
