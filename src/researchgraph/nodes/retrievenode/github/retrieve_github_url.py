import re
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
            return [f"https://{url}" if not url.startswith("http") else url for url in matches]
        except Exception as e:
            print(f"Error extracting GitHub URL: {e}")
            return []

    def execute(self, state) -> dict:
        paper_text = getattr(state, self.input_key[0])
        github_url = getattr(state, self.output_key[0])
        github_url = self._extract_github_url_from_text(paper_text)

        return {self.output_key[0]: github_url}
