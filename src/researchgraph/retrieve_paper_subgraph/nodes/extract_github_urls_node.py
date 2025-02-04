import re
import requests


class ExtractGithubUrlsNode:
    def __init__(
        self,
    ):
        pass

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

    def execute(self, paper_text) -> list[str]:
        github_url = self._extract_github_url_from_text(paper_text)

        return github_url


if __name__ == "__main__":
    extract_github_url_node = ExtractGithubUrlsNode()
    paper_text = ""
    github_urls = extract_github_url_node.execute(paper_text)
    print(github_urls)
