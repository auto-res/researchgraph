import json
from semanticscholar import SemanticScholar
from pydantic import BaseModel, ValidationError, validate_call
from researchgraph.core.node import Node
from researchgraph.nodes.retrievenode.base.paper_search import PaperSearch
from typing import Optional
from datetime import datetime, timedelta

class SemanticScholarResponse(BaseModel):
    arxiv_url: str
    paper_title: str
    authors: list[dict]
    publication_date: str
    journal: Optional[str] = None
    doi: Optional[str] = None
    externalIds: Optional[dict] = None


class SemanticScholarNode(Node, PaperSearch):
    @validate_call
    def __init__(
        self,
        input_key: list[str],
        output_key: list[str],
        num_retrieve_paper: int, 
        period: Optional[int] = None
    ):
        super().__init__(input_key, output_key)
        self.num_retrieve_paper = num_retrieve_paper
        self.period = period
        if period:
            start_time = (datetime.now() - timedelta(days=period)).strftime("%Y-%m-%d")
            end_time = datetime.now().strftime("%Y-%m-%d")
            self.publicationDateOrYear = f"{start_time}:{end_time}"
        else:
            self.publicationDateOrYear = None
    
    def search_paper(self, queries: list[str]) -> list[dict]:
        sch = SemanticScholar()
        search_results = []
        for query in queries:
            search_params = {
                "query": query, 
                "limit": self.num_retrieve_paper, 
            }
            if self.publicationDateOrYear:
                search_params["publicationDateOrYear"] = self.publicationDateOrYear
            results = sch.search_paper(**search_params)
            
            for result in results:
                    try:
                        arxiv_id = result.get("externalIds", {}).get("ArXiv") if result.get("externalIds") else None
                        arxiv_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None
                        if arxiv_url:
                            validated_result = SemanticScholarResponse(
                                arxiv_url=arxiv_url, 
                                paper_title=result.get("title", "Unknown Title"),
                                authors=result.get("authors", []),
                                publication_date=result.get("publicationDate", "Unknown date"),
                                journal=result.get("journal", None),
                                doi=result.get("doi", None),
                                externalIds=result.get("externalIds", None), 
                            )
                            search_results.append(validated_result.model_dump())
                    except ValidationError as e:
                        print(f"Validation error for item {result}: {e}")
                        continue
        return search_results

    def execute(self, state) -> dict:
        queries = getattr(state, self.input_key[0])
        search_results = self.search_paper(queries)

        return {
            self.output_key[0]: search_results, 
        }
