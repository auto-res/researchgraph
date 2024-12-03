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
        input_key: list[str],
        output_key: list[str],
        save_dir: str,
    ):
        super().__init__(input_key, output_key)
        self.save_dir = save_dir

    def execute(self, state) -> dict:
        arxiv_url = state[self.input_key[0]]
        arxiv_id = re.sub(r"^https://arxiv\.org/abs/", "", arxiv_url)

        text_path = os.path.join(self.save_dir, f"{arxiv_id}.txt")
        pdf_path = os.path.join(self.save_dir, f"{arxiv_id}.pdf")

        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as text_file:
                full_text = text_file.read()
            logging.info(f"Loaded text from {text_path}")

        else:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            try:
                response = requests.get(pdf_url, stream=True)
                response.raise_for_status()
                with open(pdf_path, "wb") as file:
                    shutil.copyfileobj(response.raw, file)
            except RequestException as e:
                logging.error(f"Failed to download {pdf_url}: {e}")
                return {self.output_key: None}

            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            full_text = "".join(page.page_content.replace("\n", "") for page in pages)
            with open(text_path, "w", encoding="utf-8") as text_file:
                text_file.write(full_text)

        return {
            self.output_key[0]: full_text,
        }
