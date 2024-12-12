import os
import re
import shutil
import requests
import logging
from requests.exceptions import RequestException
from langchain_community.document_loaders import PyPDFLoader

from researchgraph.core.node import NodeExecutionError
from researchgraph.nodes.retrievenode.base.base import BaseRetrieveNode


class RetrievearXivTextNode(BaseRetrieveNode):
    def __init__(
        self,
        input_key: list[str],
        output_key: list[str],
        save_dir: str,
        max_retries: int = 3,
        retry_delay: int = 1,
    ):
        super().__init__(input_key, output_key, max_retries, retry_delay)
        self.save_dir = save_dir

    def _download_pdf(self, pdf_url: str, pdf_path: str) -> None:
        def download_operation():
            response = requests.get(pdf_url, stream=True)
            response.raise_for_status()
            with open(pdf_path, "wb") as file:
                shutil.copyfileobj(response.raw, file)

        self.execute_with_retry(download_operation)

    def execute(self, state) -> dict:
        arxiv_url = state[self.input_key[0]]
        arxiv_id = re.sub(r"^https://arxiv\.org/abs/", "", arxiv_url)

        text_path = os.path.join(self.save_dir, f"{arxiv_id}.txt")
        pdf_path = os.path.join(self.save_dir, f"{arxiv_id}.pdf")

        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as text_file:
                full_text = text_file.read()
            self.logger.info(f"Loaded text from {text_path}")

        else:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            try:
                self._download_pdf(pdf_url, pdf_path)
                loader = PyPDFLoader(pdf_path)
                pages = loader.load_and_split()
                full_text = "".join(page.page_content.replace("\n", "") for page in pages)
                with open(text_path, "w", encoding="utf-8") as text_file:
                    text_file.write(full_text)
            except Exception as e:
                self.logger.error(f"Failed to process {pdf_url}: {str(e)}")
                raise NodeExecutionError(
                    self.__class__.__name__,
                    f"Failed to process paper: {str(e)}",
                    original_error=e
                )

        return {
            self.output_key[0]: full_text,
        }
