import os
import re
import shutil
import requests
import logging
from requests.exceptions import RequestException

from typing import Any, TypedDict
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger("researchgraph")


class State(TypedDict):
    arxiv_url: str
    paper_text: str


class RetrievearXivTextNode:
    def __init__(self, input_variable, output_variable, save_dir):
        self.input_variable = input_variable
        self.output_variable = output_variable
        self.save_dir = save_dir
        print("RetrievearXivTextNode initialized")
        print(f"input: {self.input_variable}")
        print(f"output: {self.output_variable}")

    def __call__(self, state: State, config: RunnableConfig) -> dict[str, Any]:
        arxiv_url = state[self.input_variable]
        arxiv_id = re.sub(r"^https://arxiv\.org/abs/", "", arxiv_url)

        pdf_path = os.path.join(self.save_dir, f"{arxiv_id}.pdf")
        text_path = os.path.join(self.save_dir, f"{arxiv_id}.txt")

        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as text_file:
                full_text = text_file.read()

        else:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            try:
                response = requests.get(pdf_url, stream=True)
                response.raise_for_status()
                with open(pdf_path, "wb") as file:
                    shutil.copyfileobj(response.raw, file)
            except RequestException as e:
                logging.error(f"Failed to download {pdf_url}: {e}")
                return {self.output_variable: None}

            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            full_text = "".join(page.page_content.replace("\n", "") for page in pages)
            with open(text_path, "w", encoding="utf-8") as text_file:
                text_file.write(full_text)

        logger.info("---RetrievearXivTextNode---")
        logger.info(f"Full paper text: {full_text[:100]}")
        return {
            self.output_variable: full_text,
        }


if __name__ == "__main__":
    save_dir = "/workspaces/researchgraph/data"
    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "csvretriever",
        RetrievearXivTextNode(
            save_dir=save_dir, input_variable="arxiv_url", output_variable="paper_text"
        ),
    )
    graph_builder.set_entry_point("csvretriever")
    graph_builder.set_finish_point("csvretriever")
    graph = graph_builder.compile()

    memory = {
        "arxiv_url": "https://arxiv.org/abs/1604.03540v1",
    }

    graph.invoke(memory, debug=True)
