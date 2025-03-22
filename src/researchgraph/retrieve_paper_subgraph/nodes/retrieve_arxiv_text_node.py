import os
import re
import shutil
import requests
import logging
from requests.exceptions import RequestException
from langchain_community.document_loaders import PyPDFLoader


class RetrievearXivTextNode:
    def __init__(
        self,
        save_dir: str,
    ):
        self.save_dir = save_dir

    def execute(self, arxiv_url: str) -> str:
        # arxiv_url = getattr(state, self.input_key[0])
        os.makedirs(self.save_dir, exist_ok=True)

        arxiv_id = re.sub(r"^https?://arxiv\.org/abs/", "", arxiv_url)

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
            with open(text_path, "w", encoding="utf-8", errors="replace") as text_file:
                text_file.write(full_text)

        return full_text


if __name__ == "__main__":
    save_dir = "/workspaces/researchgraph/data"
    arxiv_url = "https://arxiv.org/abs/2106.06869"
    retrieve_arxiv_text_node = RetrievearXivTextNode(save_dir)
    full_text = retrieve_arxiv_text_node.execute(arxiv_url)
    print(type(full_text))
    # print(full_text)
