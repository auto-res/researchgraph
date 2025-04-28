from pydantic import BaseModel
from jinja2 import Environment

# from researchgraph.utils.openai_client import openai_client
from airas.utils.vertexai_client import vertexai_client


class LLMOutput(BaseModel):
    main_contributions: str
    methodology: str
    experimental_setup: str
    limitations: str
    future_research_directions: str


# TODO：Changed to gemini because the prompt is very long.
def summarize_paper_node(
    llm_name: str,
    prompt_template: str,
    paper_text: str,
) -> tuple[str, str, str, str, str]:
    data = {
        "paper_text": paper_text,
    }

    env = Environment()
    template = env.from_string(prompt_template)
    prompt = template.render(data)
    # TODO：OpenAI clientと統合した際に修正
    # messages = [
    #     {"role": "user", "content": f"{prompt}"},
    # ]
    response = vertexai_client(
        model_name=llm_name, message=prompt, data_model=LLMOutput
    )
    # response = openai_client(llm_name, message=messages, data_model=LLMOutput)
    # response = json.loads(response)
    main_contributions = response["main_contributions"]
    methodology = response["methodology"]
    experimental_setup = response["experimental_setup"]
    limitations = response["limitations"]
    future_research_directions = response["future_research_directions"]

    return (
        main_contributions,
        methodology,
        experimental_setup,
        limitations,
        future_research_directions,
    )


summarize_paper_prompt_base = """
You are an expert research assistant responsible for summarizing a research paper that will serve as the foundation (Research A) for further exploration and integration.

Your task is to generate a structured summary of the given research paper with a focus on:
- **Technical Contributions**: Identify the main research problem and key findings.
- **Methodology**: Describe the techniques, models, or algorithms used.
- **Experimental Setup**: Outline the datasets, benchmarks, and validation methods.
- **Limitations**: Highlight any weaknesses, constraints, or assumptions.
- **Future Research Directions**: Suggest possible extensions or new areas for research.

Below is the full text of the research paper:

```
{{ paper_text }}
```

## **Instructions:**
1. Analyze the paper based on the categories listed below.
2. Your response **must be a valid JSON object** that can be directly parsed using `json.loads()`.
3. Do not include any extra text, explanations, or formatting outside of the JSON object.
4. **If a field has no available information, set its value to `"Not mentioned"` instead of leaving it empty.**
5. Ensure that the JSON format is correct, including the use of **double quotes (`"`) for all keys and values.**
## **Output Format (JSON)**:
```json
{
    "main_contributions": "<Concise description of the main research problem and contributions>",
    "methodology": "<Brief explanation of the key techniques, models, or algorithms>",
    "experimental_setup": "<Description of datasets, benchmarks, and validation methods>",
    "limitations": "<Summary of weaknesses, constraints, or assumptions>",
    "future_research_directions": "<Potential areas for extending this research>"
}
```
"""


summarize_paper_prompt_add = """
You are an expert research assistant responsible for summarizing a research paper that will be integrated into an existing body of knowledge (Research B).

Your task is to generate a structured summary of this research paper with a focus on:
- **Technical Contributions**: Identify the main research problem and key findings.
- **Methodology**: Describe the techniques, models, or algorithms used.
- **Experimental Setup**: Outline the datasets, benchmarks, and validation methods.
- **Limitations**: Highlight any weaknesses, constraints, or assumptions.
- **Relevance to Existing Research**: Explain how this paper contributes to or extends prior work (e.g., Research A).
- **Potential Applications**: Identify practical applications of the findings.

Below is the full text of the research paper:

```
{{ paper_text }}
```

## **Instructions:**
1. Analyze the paper based on the categories listed below.
2. Your response **must be a valid JSON object** that can be directly parsed using `json.loads()`.
3. Do not include any extra text, explanations, or formatting outside of the JSON object.
4. **If a field has no available information, set its value to `"Not mentioned"` instead of leaving it empty.**
5. Ensure that the JSON format is correct, including the use of **double quotes (`"`) for all keys and values.**

## **Output Format (JSON)**:
```json
{
    "main_contributions": "<Concise description of the main research problem and contributions>",
    "methodology": "<Brief explanation of the key techniques, models, or algorithms>",
    "experimental_setup": "<Description of datasets, benchmarks, and validation methods>",
    "limitations": "<Summary of weaknesses, constraints, or assumptions>",
    "relevance_to_existing_research": "<Explanation of how this paper contributes to or extends Research A>",
    "potential_applications": "<Practical applications of this research>"
}
```
"""
