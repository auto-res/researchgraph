# %%
import os
import json
import shutil
import requests
from langchain_community.document_loaders import PyPDFLoader
from semanticscholar import SemanticScholar
from typing import Any, TypedDict
from langgraph.graph import StateGraph
from pydantic import BaseModel, ValidationError, validate_arguments


class State(TypedDict):
    keywords: str
    collection_of_papers: dict


class SemanticScholarResponse(BaseModel):
    paper_title: str
    paper_abstract: str
    authors: list[str]
    publication_date: str


class SemanticScholarNode:
    @validate_arguments
    def __init__(
        self,
        save_dir: str,
        search_variable: str,
        output_variable: str,
        num_retrieve_paper: int,
    ):
        self.save_dir = save_dir
        self.search_variable = search_variable
        self.output_variable = output_variable
        self.num_retrieve_paper = num_retrieve_paper
        print("SemanticScholarRetriever initialized")
        print(f"input: {search_variable}")
        print(f"output: {output_variable}")

    def _download_from_arxiv_id(self, arxiv_id: str) -> None:
        """Download PDF file from arXiv

        Args:
            arxiv_id (_type_): _description_
        """

        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open(os.path.join(self.save_dir, f"{arxiv_id}.pdf"), "wb") as file:
                shutil.copyfileobj(response.raw, file)
            print(f"Downloaded {arxiv_id}.pdf to {self.save_dir}")
        else:
            print(f"Failed to download {arxiv_id}.pdf")

    def _download_from_arxiv_ids(self, arxiv_ids: list[str]) -> None:
        """Download PDF files from arXiv

        Args:
            arxiv_ids (_type_): _description_
            save_dir (_type_): _description_
        """
        # save_dirが存在しない場合、ディレクトリを作成
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            shutil.rmtree(self.save_dir)
            os.makedirs(self.save_dir)

        for arxiv_id in arxiv_ids:
            self._download_from_arxiv_id(arxiv_id)

    def _convert_pdf_to_text(self, pdf_path: str) -> str:
        """Convert PDF file to text

        Args:
            pdf_path (_type_): _description_

        Returns:
            _type_: _description_
        """

        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        content = ""
        for page in pages[:20]:
            content += page.page_content

        return content

    def __call__(self, state: State) -> dict[str, Any]:
        """Retriever

        Args:
            state (_type_): _description_
        """
        keywords_list = json.loads(state[self.search_variable])
        # keywords_list = [keywords_list[: self.num_keywords]]
        sch = SemanticScholar()

        all_search_results = []
        for search_term in keywords_list:
            results = sch.search_paper(search_term, limit=self.num_retrieve_paper)

            # Validate each result using Pydantic
            validated_results = []
            for item in results.items:
                try:
                    validated_result = SemanticScholarResponse(
                        paper_title=getattr(item, "title", "Unknown Title"),
                        paper_abstract=getattr(item, "abstract", "No abstract available."),
                        authors=getattr(item, "authors", []),
                        publication_date=getattr(item, "publicationDate", "Unknown date")
                    )
                    validated_results.append(validated_result.dict())
                except ValidationError as e:
                    print(f"Validation error for item {item}: {e}")

            all_search_results.append(validated_results)

        DOI_ids = [
            item["externalIds"]
            for results in all_search_results
            for item in results.items
        ]
        arxiv_ids = [item["ArXiv"] for item in DOI_ids if "ArXiv" in item]

        self._download_from_arxiv_ids(arxiv_ids[: self.num_retrieve_paper])

        paper_list_dict = {}
        for idx, filename in enumerate(os.listdir(self.save_dir)):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(self.save_dir, filename)
                paper_content = self._convert_pdf_to_text(pdf_path)
                paper_key = f"paper_{idx+1}"
                paper_list_dict[paper_key] = paper_content

        return {self.output_variable: paper_list_dict}


if __name__ == "__main__":
    save_dir = "/workspaces/researchgraph/data"
    search_variable = "keywords"
    output_variable = "collection_of_papers"

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "semanticscholarretriever",
        SemanticScholarNode(
            save_dir=save_dir,
            search_variable=search_variable,
            output_variable=output_variable,
            num_retrieve_paper=3,
        ),
    )
    graph_builder.set_entry_point("semanticscholarretriever")
    graph_builder.set_finish_point("semanticscholarretriever")
    graph = graph_builder.compile()

    memory = {"keywords": '["Grokking"]'}

    graph.invoke(memory, debug=True)
