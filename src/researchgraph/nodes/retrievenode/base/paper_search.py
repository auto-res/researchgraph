from abc import ABC, abstractmethod
from typing import Any

class PaperSearch(ABC):
    """
    Abstract class for paper search node
    """

    @abstractmethod
    def search_paper(self, keywords: list[str], num_retrieve_paper: int) -> list[dict[str, Any]]:
        """
        Method to search papers with specified keywords
        """
        pass


    def download_from_arxiv_ids(self, arxiv_ids: list[str]) -> None:
        """
        Method to download a paper PDF by specified arXiv ID
        """
        pass


    def convert_pdf_to_text(self, pdf_path: str) -> str:
        """
        Method to convert a PDF to text
        """
        pass
