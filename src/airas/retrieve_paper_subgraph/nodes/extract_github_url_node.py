import re
import requests
from jinja2 import Environment

# from researchgraph.utils.openai_client import openai_client
from airas.utils.vertexai_client import vertexai_client
from airas.retrieve_paper_subgraph.prompt.extract_github_url_node_prompt import (
    extract_github_url_node_prompt,
)
from pydantic import BaseModel
from logging import getLogger

logger = getLogger(__name__)


class LLMOutput(BaseModel):
    index: int | None


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
            logger.warning(f"Error extracting GitHub URL: {e}")
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
    ) -> int | None:
        env = Environment()
        template = env.from_string(extract_github_url_node_prompt)
        data = {
            "paper_summary": paper_summary,
            "extract_github_url_list": extract_github_url_list,
        }
        prompt = template.render(data)
        # TODO：OpenAI clientと統合した際に修正
        #         message = [
        #             {
        #                 "role": "system",
        #                 "content": """\
        # # Task
        # You carefully read the contents of the “Paper Outline” and select one GitHub link from the “GitHub URLs List” that you think is most relevant to the contents.

        # # Constraints
        # - Output the index number corresponding to the selected GitHub URL.
        # - Be sure to select only one GiHub URL.
        # - If there is no related GitHub link, output None.""",
        #             },
        #             {
        #                 "role": "user",
        #                 "content": f"""\
        # # Paper Outline
        # {paper_summary}

        # # GitHub URLs List
        # {extract_github_url_list}""",
        #             },
        #         ]

        response = vertexai_client(
            model_name=self.llm_name, message=prompt, data_model=LLMOutput
        )
        # response = openai_client(self.llm_name, message=message, data_model=LLMOutput)
        if response is None:
            logger.warning("Error: No response from LLM.")
            return None
        else:
            # response = json.loads(response)
            return response["index"]

    def execute(self, paper_full_text: str, paper_summary: str) -> str:
        extract_github_url_list = self._extract_github_url_from_text(paper_full_text)
        if not extract_github_url_list:
            return ""
        index = self._extract_related_github_url(paper_summary, extract_github_url_list)
        if index is None:
            return ""
        elif 0 <= index <= len(extract_github_url_list) - 1:
            return extract_github_url_list[index]
        else:
            logger.warning(
                "An index outside the range of extract_github_url_list was selected."
            )
            return ""


if __name__ == "__main__":
    extract_github_url_node = ExtractGithubUrlNode(
        llm_name="gemini-2.0-flash-001",
    )
    paper_text = "aaa"
    paper_summary = "bbb"
    github_urls = extract_github_url_node.execute(
        paper_full_text=paper_text,
        paper_summary=paper_summary,
    )
    print(github_urls)
