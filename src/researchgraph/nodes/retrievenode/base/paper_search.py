from abc import ABC, abstractmethod
from typing import Any, Optional

class PaperSearch(ABC):
    """
    Abstract class for paper search node
    """

    @abstractmethod
    def search_paper(self, queries: list[str], num_retrieve_paper: int, period: Optional[str] = None) -> list[dict[str, Any]]:
        """
        Method to search papers with specified keywords
        """
        pass