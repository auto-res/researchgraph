from researchgraph.retrieve_paper_subgraph.nodes.search_api.arxiv_api_node import (
    ArxivNode,
)
# from researchgraph.nodes.retrievenode.semantic_scholar.semantic_scholar import SemanticScholarNode


class SearchPapersNode:
    def __init__(
        self,
        period_days: int = 40,
        num_retrieve_paper: int = 5,
        api_type: str = "arxiv",
    ):
        self.period_days = period_days
        self.num_retrieve_paper = num_retrieve_paper
        self.api_type = api_type

    def execute(self, queries: list[str]) -> list[dict]:
        if self.api_type == "arxiv":
            paper_search = ArxivNode(
                num_retrieve_paper=self.num_retrieve_paper,
                period_days=self.period_days,
            )
        # elif self.api_type == "semanticscholar":
        #     paper_search = SemanticScholarNode(
        #         input_key=self.input_key,
        #         output_key=self.output_key,
        #         num_retrieve_paper=self.num_retrieve_paper,
        #         period_days=self.period_days,
        #     )
        else:
            raise ValueError(f"Invalid api_type: {self.api_type}")

        try:
            search_results = paper_search.execute(queries)
            if search_results is None:
                raise Exception("No search results found")
            # search_results = search_results.get("search_results")
        except Exception as e:
            raise Exception(f"Error during paper search: {e}")

        return search_results


if __name__ == "__main__":
    queries = [
        "graph neural networks",
        "transformer",
    ]
    search_papers_node = SearchPapersNode()
    search_results = search_papers_node.execute(queries)
    print(search_results)
