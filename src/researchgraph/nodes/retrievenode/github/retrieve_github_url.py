import re
import requests
from researchgraph.core.node import Node


class RetrieveGithubUrlNode(Node):
    def __init__(
        self,
        input_key: list[str],
        output_key: list[str],

    ):
        super().__init__(input_key, output_key)

    def _extract_github_url_from_text(self, content: str) -> list[str]:
        try:
            matches = re.findall(r"https?://github\.com/[\w\-\_]+/[\w\-\_]+", content)
            valid_urls = []
            for url in matches:
                url = url.replace("http://", "https://")
                if self._is_valid_github_url(url):
                    valid_urls.append(url)
            return valid_urls
        except Exception as e:
            print(f"Error extracting GitHub URL: {e}")
            return []
    
    def _is_valid_github_url(self, url: str) -> bool:
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error checking GitHub URL {url}: {e}")
            return False

    def execute(self, state) -> dict:
        paper_text = getattr(state, self.input_key[0])
        github_url = getattr(state, self.output_key[0])
        github_url = self._extract_github_url_from_text(paper_text)

        return {self.output_key[0]: github_url}
