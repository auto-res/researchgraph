from pydantic import BaseModel
from jinja2 import Environment
import json
from researchgraph.utils.openai_client import openai_client


class LLMOutput(BaseModel):
    generated_citation_queries: list[str]


def generate_citation_queries(
    llm_name: str,
    prompt_template: str,
    contexts: list[str],
    reasons: list[str],
) -> list[str] | None:
    data = {
        "citation_items": [
            {"context": ctx, "reason": reason}
            for ctx, reason in zip(contexts, reasons)
        ]
    }

    env = Environment()
    template = env.from_string(prompt_template)
    prompt = template.render(data)

    messages = [
        {"role": "user", "content": prompt},
    ]
    response = openai_client(llm_name, message=messages, data_class=LLMOutput)
    response = json.loads(response)
    if not response:
        return None

    generated_citation_queries = response["generated_citation_queries"]

    return generated_citation_queries

generate_citation_queries_prompt = """
You are an expert research assistant.

Your task is to generate one short and precise search query for **each citation context**.

Below are citation items with their context and the reason they require a citation:
{% for item in citation_items %}
- Context: {{ item.context }}
  Reason: {{ item.reason }}
{% endfor %}

## Instructions:
1. Analyze each context and reason carefully.
2. For each one, output exactly one concise and relevant search query (2–7 words).
3. Focus on academic topics, model names, technical methods, or theoretical ideas.
4. Do NOT include citations related to real-world domains (e.g., healthcare, business).
5. Output the result as a JSON list of strings (queries), in the same order as the input.

## Output Format:
```json
[
  "query for first citation",
  "query for second citation",
  ...
]
"""

if __name__ == "__main__":
    contexts = [
        "Deep learning has significantly improved the performance of computer vision tasks.",
        "Recent advancements include transformer-based models in vision as well.",
    ]
    reasons = [
        "This claim refers to well-established findings and should be supported by prior work.",
        "Transformer models are a major shift in vision modeling and require attribution.",
    ]
    llm_name = "o3-mini-2025-01-31"

    generated_citation_queries = generate_citation_queries(
        llm_name=llm_name,
        prompt_template=generate_citation_queries_prompt,
        contexts=contexts,
        reasons=reasons,
    )
    if not generated_citation_queries:
        print("Failed to generate citation queries.")
        exit(1)

    print("=== Generated Queries ===")
    for i, q in enumerate(generated_citation_queries):
        print(f"[{i+1}] {q}")
