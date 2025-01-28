from researchgraph.core.node import Node
from researchgraph.nodes.retrievenode.arxiv_api.arxiv_api_node import ArxivNode
from researchgraph.nodes.retrievenode.semantic_scholar.semantic_scholar import SemanticScholarNode

class RetrievePaperNode(Node):
    def __init__(
        self,
        input_key: list[str],
        output_key: list[str],
        period_days: int = 7,
        num_retrieve_paper: int = 5, 
        api_type: str = "arxiv", 
    ):
        super().__init__(input_key, output_key)
        self.period_days = period_days
        self.num_retrieve_paper = num_retrieve_paper
        self.api_type = api_type

    def execute(self, state) -> dict:
        queries = getattr(state, self.input_key[0])
        search_results = getattr(state, self.output_key[0])
        if self.api_type == "arxiv":
            paper_search = ArxivNode(
                input_key=self.input_key, 
                output_key=self.output_key, 
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
            search_results = paper_search.execute(state)
            if search_results is None:
                raise Exception("No search results found")
            search_results = search_results.get(self.output_key[0])
        except Exception as e:
            raise Exception(f"Error during paper search: {e}")
        
        return {
            self.output_key[0]: search_results, 
        }

