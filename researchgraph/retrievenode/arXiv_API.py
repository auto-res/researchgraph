import os
import json
import shutil
import requests
from langchain_community.document_loaders import PyPDFLoader
import arxiv
from pydantic import BaseModel, ValidationError, validate_arguments
from typing import Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph


class State(TypedDict):
    keywords: list[str]
    collection_of_papers: dict


class ArxivResponse(BaseModel):
    title: str
    arxiv_id: str
    authors: list[str]
    abstract: str
    published_date: str


class ArxivNode:
    @validate_arguments
    def __init__(
            self, 
            save_dir: str, 
            search_variable: str, 
            output_variable: str, 
            num_keywords: int, 
            num_retrieve_paper: int
        ):
        self.save_dir = save_dir
        self.search_variable = search_variable
        self.output_variable = output_variable
        self.num_keywords = num_keywords
        self.num_retrieve_paper = num_retrieve_paper

        print("ArxivNode initialized")
        print(f"input: {search_variable}")
        print(f"output: {output_variable}")

    def download_from_arxiv_id(self, arxiv_id):
        """Download PDF file from arXiv

        Args:
            arxiv_id (str): The arXiv identifier of the paper
        """

        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {arxiv_id}.pdf: {e}")
            return

        with open(os.path.join(self.save_dir, f"{arxiv_id}.pdf"), 'wb') as file:
            shutil.copyfileobj(response.raw, file)
        print(f"Downloaded {arxiv_id}.pdf to {self.save_dir}")
 
    def download_from_arxiv_ids(self, arxiv_ids: list[str]):
        """Download PDF files from arXiv

        Args:
            arxiv_ids (list): List of arXiv identifiers
        """
        # Create the save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else :
            # Clear the directory if it already exists
            shutil.rmtree(self.save_dir)
            os.makedirs(self.save_dir)

        for arxiv_id in arxiv_ids:
            self.download_from_arxiv_id(arxiv_id)

    def convert_pdf_to_text(self, pdf_path: str, max_pages: int = 20):
        """Convert PDF file to text

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            str: Extracted text content from the PDF
        """

        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        content = ""
        for page in pages[:max_pages]:
            content += page.page_content

        return content

    def __call__(self, state: State) -> State:
        """Retrieve papers from arXiv based on keywords

        Args:
            state (State): The current state containing keywords

        Returns:
            State: Updated state with downloaded papers
        """
        keywords_list = json.loads(state[self.search_variable])
        all_search_results = []

        client = arxiv.Client(
            num_retries=3,  # 再試行の設定
            page_size=self.num_retrieve_paper  # ページサイズを設定
        )

        for search_term in keywords_list:
            search = arxiv.Search(
                query=search_term,
                max_results=self.num_retrieve_paper,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )
            
            results = list(client.results(search))
            all_search_results.extend(results)

        print(f"all_search_results {len(all_search_results)}")

        validated_results = []
        for result in all_search_results:
            try:
                validated_result = ArxivResponse(
                    title=result.title,
                    arxiv_id=result.get_short_id(),
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    published_date=str(result.published)
                )
                validated_results.append(validated_result)
            except ValidationError as e:
                print(f"Validation error for item {result}: {e}")

        arxiv_ids = [result.arxiv_id for result in validated_results]
        self.download_from_arxiv_ids(arxiv_ids)

        if self.output_variable not in state:
            state[self.output_variable] = {}

        # Process downloaded PDFs
        for idx, filename in enumerate(os.listdir(self.save_dir)):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.save_dir, filename)
                paper_content = self.convert_pdf_to_text(pdf_path)
                paper_key = f"paper_{idx+1}"
                state[self.output_variable][paper_key] = paper_content
        return state


if __name__ == "__main__":
    save_dir = "/workspaces/researchchain/data"
    search_variable = "keywords"
    output_variable = "collection_of_papers"

    memory = {
        "keywords": '["Grokking", "Separability"]'
    }

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "arxivretriever",
        ArxivNode(
            save_dir=save_dir,
            search_variable=search_variable,
            output_variable=output_variable,
            num_keywords=2,
            num_retrieve_paper=5,
            # num_retrieve_paper=1,
        )
    )
    graph_builder.set_entry_point("arxivretriever")
    graph_builder.set_finish_point("arxivretriever")
    graph = graph_builder.compile()

    memory = {"keywords": '["Grokking", "Separability"]'}

    graph.invoke(memory, debug=True)
