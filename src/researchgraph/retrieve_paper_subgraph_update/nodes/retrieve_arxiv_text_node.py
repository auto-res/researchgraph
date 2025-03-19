import os
import re
import shutil
import requests
import logging
from requests.exceptions import RequestException
from langchain_community.document_loaders import PyPDFLoader


class RetrieveArxivTextNode:
    def __init__(
        self,
        save_dir: str,
    ):
        self.save_dir = save_dir
        # papers ディレクトリのパスを作成
        self.papers_dir = os.path.join(save_dir, "papers")
        # ディレクトリが存在することを確認
        os.makedirs(self.papers_dir, exist_ok=True)

    def execute(self, arxiv_url: str) -> str:
        arxiv_id = re.sub(r"^https?://arxiv\.org/abs/", "", arxiv_url)

        # papers ディレクトリ内にファイルを保存
        text_path = os.path.join(self.papers_dir, f"{arxiv_id}.txt")
        pdf_path = os.path.join(self.papers_dir, f"{arxiv_id}.pdf")

        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as text_file:
                full_text = text_file.read()
            logging.info(f"Loaded text from {text_path}")

        else:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            try:
                print(f"    Downloading PDF from {pdf_url}")
                response = requests.get(pdf_url, stream=True)
                response.raise_for_status()
                with open(pdf_path, "wb") as file:
                    shutil.copyfileobj(response.raw, file)
                print(f"    PDF saved to {pdf_path}")
            except RequestException as e:
                logging.error(f"Failed to download {pdf_url}: {e}")
                return "Failed to download PDF"

            print(f"    Extracting text from PDF")
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            full_text = "".join(page.page_content.replace("\n", "") for page in pages)
            with open(text_path, "w", encoding="utf-8") as text_file:
                text_file.write(full_text)
            print(f"    Text saved to {text_path}")

        return full_text


if __name__ == "__main__":
    save_dir = "/workspaces/researchgraph/data"
    arxiv_url = "https://arxiv.org/abs/2106.06869"
    retrieve_arxiv_text_node = RetrieveArxivTextNode(save_dir)
    full_text = retrieve_arxiv_text_node.execute(arxiv_url)
    print(type(full_text))
    # print(full_text)
