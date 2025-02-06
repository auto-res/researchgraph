import re
import json
import requests
from litellm import completion

from pydantic import BaseModel


class LLMOutput(BaseModel):
    index: str


class ExtractGithubUrlNode:
    def __init__(
        self,
        llm_name: str,
    ):
        self.llm_name = llm_name

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

    def _extract_related_github_url(
        self, paper_summary: str, extract_github_url_list: list[str]
    ) -> str:
        messate = [
            {
                "role": "system",
                "content": """
# Task
You carefully read the contents of the “Paper Outline” and select one GitHub link from the “GitHub URLs List” that you think is most relevant to the contents.

# Constraints
- Output the index number corresponding to the selected GitHub URL.
- Be sure to select only one GiHub URL.
- If there is no related GitHub link, output an empty string.""",
            },
            {
                "role": "user",
                "content": f"""
# Paper Outline
{paper_summary}
      
# GitHub URLs List
{extract_github_url_list}""",
            },
        ]

        response = completion(
            model=self.llm_name,
            messages=messate,
            response_format=LLMOutput,
        )
        list_index_str = json.loads(response.choices[0].message.content)["index"]
        return list_index_str

    def execute(self, paper_full_text: str, paper_summary: str) -> str:
        extract_github_url_list = self._extract_github_url_from_text(paper_full_text)
        if not extract_github_url_list:
            return ""
        list_index_str = self._extract_related_github_url(
            paper_summary, extract_github_url_list
        )
        if not list_index_str:
            return ""

        list_index = int(list_index_str)
        if list_index >= len(extract_github_url_list):
            print("extract_github_url_listの範囲外のindexが選択されました")
            return ""

        return extract_github_url_list[list_index]


if __name__ == "__main__":
    extract_github_url_node = ExtractGithubUrlNode()
    paper_text = ""
    github_urls = extract_github_url_node.execute(paper_text)
    print(github_urls)
