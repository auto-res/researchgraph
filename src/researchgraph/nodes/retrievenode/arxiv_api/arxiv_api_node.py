import requests
import feedparser
import pytz
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel, ValidationError, Field
from researchgraph.core.node import Node

class ArxivResponse(BaseModel):
    arxiv_id: str
    arxiv_url: str
    title: str
    authors: list[str]
    published_date: str
    summary: str = Field(default="No summary")

class ArxivNode(Node):
    def __init__(
        self,
        input_key: list[str],
        output_key: list[str],
        num_retrieve_paper: int = 5,
        period_days: Optional[int] = 7
    ):
        super().__init__(input_key, output_key)
        self.num_retrieve_paper = num_retrieve_paper
        self.period_days = period_days
        self.start_indices: dict[str, int] = {} #TODO: stateに持たせる？

    def _build_arxiv_query(self, query: str) -> str:
        now_utc = datetime.now(pytz.utc)
        if self.period_days is None:

            return f"all:{query}"

        from_date = now_utc - timedelta(days=self.period_days)
        from_str = from_date.strftime("%Y-%m-%d")
        to_str = now_utc.strftime("%Y-%m-%d")
        return f"(all:{query}) AND submittedDate:[{from_str} TO {to_str}]"

    def _validate_and_convert(self, entry) -> Optional[ArxivResponse]:
        try:
            paper = ArxivResponse(
                arxiv_id = entry.id.split("/")[-1], 
                arxiv_url=entry.id,
                title=entry.title or "No Title",
                authors = [a.name for a in entry.authors] if hasattr(entry, "authors") else [], 
                published_date=entry.published if hasattr(entry, "published") else "Unknown date",
                summary=entry.summary if hasattr(entry, "summary") else "No summary"
            )
            return paper
        except ValidationError as e:
            print(f"Validation error: {e}")
            return None

    def search_paper(self, query: str) -> list[ArxivResponse]:
        base_url = "http://export.arxiv.org/api/query"
        search_query = self._build_arxiv_query(query)
        start_index = self.start_indices.get(query, 0)

        params = {
            "search_query": search_query,
            "start": start_index,
            "max_results": self.num_retrieve_paper,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        try:
            response = requests.get(base_url, params=params, timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            print(f"Error fetching from arXiv API: {exc}")
            return []

        feed = feedparser.parse(response.text)

        validated_list = []
        for entry in feed.entries:
            paper = self._validate_and_convert(entry)
            if paper:
                validated_list.append(paper.model_dump())

        if len(validated_list) == self.num_retrieve_paper:
            if query not in self.start_indices:
                self.start_indices[query] = 0
            self.start_indices[query] += self.num_retrieve_paper

        return validated_list

    def execute(self, state) -> list[dict]:
        queries = getattr(state, self.input_key[0], [])
        if not queries or not isinstance(queries, list):
            print(f"No valid queries found in state[{self.input_key[0]}]. Return empty.")
            return {self.output_key[0]: []}

        all_papers = []
        for q in queries:
            results = self.search_paper(q)
            all_papers.extend(results)
        return {
            self.output_key[0]: all_papers, 
        }
