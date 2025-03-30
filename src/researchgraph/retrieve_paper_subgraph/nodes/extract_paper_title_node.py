import json
from jinja2 import Environment
from litellm import completion
from pydantic import BaseModel
from typing import Optional


class LLMOutput(BaseModel):
    paper_titles: str


def extract_paper_title_node(
    llm_name: str, 
    queries: list, 
    scraped_results: list, 
    max_retries: int = 3
) -> Optional[list[str]]:

    env = Environment()
    template = env.from_string(extract_title_prompt)

    aggregated_titles = []
    for result in scraped_results:
        data = {
            "queries": queries,
            "result": result
        }
        prompt = template.render(data)

        for attempt in range(max_retries):
            try:
                response = completion(
                    model=llm_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    response_format=LLMOutput,
                )
                structured_output = json.loads(response.choices[0].message.content)
                titles_str = structured_output["paper_titles"]
                titles_list = [title.strip() for title in titles_str.split('\n') if title.strip()]
                print(f"Extracted paper titles: {titles_list}")
                aggregated_titles.extend(titles_list)
                break
            except Exception as e:
                print(f"[Attempt {attempt+1}/{max_retries}] Error calling LLM: {e}")
        else:
            print("Exceeded maximum retries for one of the scraped results.")
            return None
    return aggregated_titles

extract_title_prompt ="""
You are an expert at extracting research paper titles from web content. 
We have the following information:
    - Queries: {{ queries }}

Below is a block of markdown content from a research paper listing page. 
Your tasks are:
1. Identify only the titles of research papers within the markdown. 
These may appear as the text inside markdown links (for example, text enclosed in ** or within [ ] if the link text represents a title).
2. Order the extracted titles in descending order of relevance to the Query (i.e., most relevant first).
3. Output the extracted titles as a single plain text string, with each title separated by a newline character.
4. Return your answer in JSON format with a single key "paper_titles". Do not include any additional commentary or formatting.
Content:
<result>
{{ result }}
</result>
"""

if __name__ == "__main__":
    llm_name = "gpt-4o-mini-2024-07-18"
    queries = ["deep leanning"]
    scraped_results = [
    "# ICLR 2024 - Deep Learning Advances\n\nThis paper discusses recent advances in deep learning architectures and training techniques...",
    "# ICLR 2024 - Neural Networks for Vision\n\nIn this study, novel convolutional neural network designs are introduced to improve image recognition..."
    ]
    extracted_paper_titles = extract_paper_title_node(
        llm_name=llm_name, 
        queries=queries,
        scraped_results=scraped_results
    )
    print(f"Extracted paper titles: {extracted_paper_titles}")