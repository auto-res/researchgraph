from jinja2 import Environment
from pydantic import BaseModel

# import json
# from researchgraph.utils.openai_client import openai_client
from researchgraph.utils.vertexai_client import vertexai_client
from logging import getLogger

logger = getLogger(__name__)


class LLMOutput(BaseModel):
    paper_titles: list[str]


def extract_paper_title_node(
    llm_name: str,
    queries: list,
    scraped_results: list,
) -> list[str]:
    env = Environment()
    template = env.from_string(extract_title_prompt)

    aggregated_titles = []
    for result in scraped_results:
        data = {"queries": queries, "result": result}
        prompt = template.render(data)
        # TODO：OpenAI clientと統合した際に修正
        # messages = [
        #     {
        #         "role": "system",
        #         "content": "You are an expert at extracting research paper titles from web content. ",
        #     },
        #     {"role": "user", "content": prompt},
        # ]
        messages = f"""\
You are an expert at extracting research paper titles from web content. 
{prompt}
"""
        response = vertexai_client(
            model_name=llm_name, message=messages, data_model=LLMOutput
        )
        # response = openai_client(
        #     model_name=llm_name, message=messages, data_class=LLMOutput
        # )
        if response is None:
            logger.warning("Error: No response from the model.")
            break
        else:
            # response = json.loads(response)
            if "paper_titles" in response:
                titles_list = response["paper_titles"]
                aggregated_titles.extend(titles_list)
    return aggregated_titles


extract_title_prompt = """\n
"Queries" represents the user's search keywords.
"Content" is a block of markdown that lists research papers based on the user's search.
# Instructions:
- Extract only the titles of research papers from the "Content".
  - These titles may appear as the text inside markdown links (e.g., bold text or text inside square brackets [ ] if it represents a paper title).
- Sort the extracted titles in descending order of relevance to the "Queries" — meaning the most relevant titles should come first.
- Output the titles as a list of strings.
# Queries:
--------
{{ queries }}
--------
# Content:
--------
{{ result }}
--------"""

if __name__ == "__main__":
    llm_name = "gemini-2.0-flash-001"
    queries = ["deep leanning"]
    scraped_results = [
        "# ICLR 2024 - Deep Learning Advances\n\nThis paper discusses recent advances in deep learning architectures and training techniques...",
        "# ICLR 2024 - Neural Networks for Vision\n\nIn this study, novel convolutional neural network designs are introduced to improve image recognition...",
    ]
    extracted_paper_titles = extract_paper_title_node(
        llm_name=llm_name, queries=queries, scraped_results=scraped_results
    )
    print(f"Extracted paper titles: {extracted_paper_titles}")
