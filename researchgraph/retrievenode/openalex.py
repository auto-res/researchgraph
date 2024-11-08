# %%
import os
import json
import shutil
import requests
import re
import pyalex
from langchain_community.document_loaders import PyPDFLoader
from datetime import datetime
from typing import Any, TypedDict
from pydantic import BaseModel, ValidationError, validate_arguments
from langgraph.graph import StateGraph


class State(TypedDict):
    keywords: list[str]
    collection_of_papers: Any
    

class OpenAlexResponse(BaseModel):
    paper_abstract: str
    author: str
    public_date: datetime


class OpenAlexNode:
    @validate_arguments
    def __init__(
        self,
        save_dir: str,
        search_variable: str,
        output_variable: str,
        num_keywords: int,
        num_retrieve_paper: int,
    ):
        self.save_dir = save_dir
        self.search_variable = search_variable
        self.output_variable = output_variable
        self.num_keywords = num_keywords
        self.num_retrieve_paper = num_retrieve_paper
        print("OpenAlexRetriever initialized")
        print(f"input: {search_variable}")
        print(f"output: {output_variable}")

    def download_from_arxiv_id(self, arxiv_id: str) -> None:
        """Download PDF file from arXiv

        Args:
            arxiv_id (_type_): _description_
            save_dir (_type_): _description_
        """

        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open(os.path.join(self.save_dir, f"{arxiv_id}.pdf"), "wb") as file:
                shutil.copyfileobj(response.raw, file)
            print(f"Downloaded {arxiv_id}.pdf to {self.save_dir}")
        else:
            print(f"Failed to download {arxiv_id}.pdf")

    def download_from_arxiv_ids(self, arxiv_ids: list[str]) -> None:
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
            self.download_from_arxiv_id(arxiv_id)

    def convert_pdf_to_text(self, pdf_path):
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


    def __call__(self, state: State) -> Any:
        """Retriever

        Args:
            state (_type_): _description_
        """
        keywords_list = json.loads(state[self.search_variable])
        keywords_list = [keywords_list[: self.num_keywords]]

        # 2012年以降を検索対象にする
        works = pyalex.Works().filter(publication_year=">2011", is_oa=True)

        all_search_results = []
        for search_term in keywords_list:
            results = works.search(search_term).get(
                page=1, per_page=self.num_retrieve_paper
            )

            # Validate each result using Pydantic
            validated_results = []
            for item in results:
                try:
                    validated_result = OpenAlexResponse(
                        paper_abstract=item.get("abstract", ""),
                        author=item.get("author", "Unknown"),
                        public_date=datetime.strptime(item.get("publication_date", "1970-01-01"), "%Y-%m-%d"),
                    )
                    validated_results.append(validated_result)
                except ValidationError as e:
                    print(f"Validation error for item {item}: {e}")

            all_search_results.append(validated_results)

        def _get_arxiv_id_from_url(url: str) -> str | None:
            match = re.search(r"\d{4}\.\d{5}", url)
            if match:
                return match.group()

        for results in all_search_results:
            arxiv_ids = []

            for item in results:
                print(item["title"])
                print(item["id"])

                if "arxiv" not in item["indexed_in"]:
                    continue
                ind_loc = item["indexed_in"].index("arxiv")

                arxiv_url = item["locations"][ind_loc]["landing_page_url"]
                arxiv_id = _get_arxiv_id_from_url(arxiv_url)
                if arxiv_id is None:
                    continue

                arxiv_ids.append(arxiv_id)
            self.download_from_arxiv_ids(arxiv_ids[: self.num_retrieve_paper])

        return {
            self.output_variable: {
                f"paper_{idx + 1}": {
                    "full_text": self.convert_pdf_to_text(
                        os.path.join(self.save_dir, filename)
                    )
                }
                for idx, filename in enumerate(os.listdir(self.save_dir))
                if filename.endswith(".pdf")
            }
        }


if __name__ == "__main__":
    save_dir = "/workspaces/researchgraph/data"
    search_variable = "keywords"
    output_variable = "collection_of_papers"

    memory = {"keywords": '["Grokking"]'}

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "openalexretriever",
        OpenAlexNode(
            save_dir=save_dir,
            search_variable=search_variable,
            output_variable=output_variable,
            num_keywords=1,
            num_retrieve_paper=3,
        ),
    )
    graph_builder.set_entry_point("openalexretriever")
    graph_builder.set_finish_point("openalexretriever")
    graph = graph_builder.compile()

    memory = {"keywords": '["Grokking"]'}

    graph.invoke(memory)
