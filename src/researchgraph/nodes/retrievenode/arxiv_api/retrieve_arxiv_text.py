import os
import re
import shutil
import requests
import logging
from requests.exceptions import RequestException
from langchain_community.document_loaders import PyPDFLoader

from researchgraph.core.node import Node


class RetrievearXivTextNode(Node):
    def __init__(
        self,
        input_variable: list[str],
        output_variable: list[str],
        save_dir: str,
    ):
        super().__init__(input_variable, output_variable)
        self.save_dir = save_dir

    def execute(self, state) -> dict:
        arxiv_url = state[self.input_variable[0]]
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

        return {
            self.output_variable[0]: full_text,
        }
