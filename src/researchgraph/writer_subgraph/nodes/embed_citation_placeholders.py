import json
from pydantic import BaseModel
from jinja2 import Environment
from researchgraph.utils.openai_client import openai_client

class LLMOutput(BaseModel):
    text_with_citations: str
    placeholder_ids: list[str]
    surrounding_contexts: list[str]
    reasons: list[str]


def embed_citation_placeholders(
    llm_name: str,
    prompt_template: str,
    paper_content: str,
) -> tuple[str, list[str], list[str], list[str]] | None:
    data = {
        "paper_content": paper_content,
    }

    env = Environment()
    template = env.from_string(prompt_template)
    prompt = template.render(data)

    messages = [
        {"role": "user", "content": prompt},
    ]
    response = openai_client(llm_name, message=messages, data_class=LLMOutput)
    if not response:
        return None

    response = json.loads(response)
    text_with_citations = response["text_with_citations"]
    placeholder_ids = response["placeholder_ids"]
    surrounding_contexts = response["surrounding_contexts"]
    reasons = response["reasons"]

    return (
        text_with_citations, 
        placeholder_ids, 
        surrounding_contexts, 
        reasons
    )

embed_citation_prompt = """
You are an expert academic writing assistant.

Your task is to insert citation placeholders (`\\cite{auto:0001}`, `\\cite{auto:0002}`, ...) in appropriate places in the following research text.

Below is the paper content:
```
{{ paper_content }}
```

## Instructions:
1. Identify parts of the text that clearly rely on external knowledge, previous work, or facts that should be cited.
2. Insert citation placeholders in LaTeX style like `\\cite{auto:0001}` directly into the text at appropriate locations.
3. For each inserted placeholder, generate a metadata array with:
   - `placeholder_ids`: e.g. ["auto:0001", "auto:0002", ...]
   - `surrounding_contexts`: short passages (1-2 sentences) around each placeholder
   - `reasons`: short explanations why each part needs a citation
4. Do not insert too many citation placeholders. Keep the number of citations realistic — typically 5 to 15 for a standard research paper, depending on length and content relevance.

## Output Format:
```json
{
  "text_with_citations": "....",
  "placeholder_ids": ["auto:0001", "auto:0002"],
  "surrounding_contexts": [
    "In recent years, deep learning has shown great results in computer vision.",
    "Transformer-based models have become popular in natural language processing."
  ],
  "reasons": [
    "This section discusses a widely recognized trend and needs supporting citation.",
    "Mentions a known technique without attribution."
  ]
}
"""

if __name__ == "__main__":
    paper_content = (
        "Deep learning has significantly improved the performance of computer vision tasks. "
        "Convolutional neural networks (CNNs) are widely used for image classification. "
        "Recent advancements include transformer-based models in vision as well."
    )

    result = embed_citation_placeholders(
        llm_name="o3-mini-2025-01-31",
        prompt_template=embed_citation_prompt,
        paper_content=paper_content,
    )
    if result:
        text_with_citations, placeholder_ids, contexts, reasons = result
        print("=== Cited Text ===")
        print(text_with_citations)
        print("\n=== Placeholders ===")
        for pid, ctx, reason in zip(placeholder_ids, contexts, reasons):
            print(f"{pid}: {ctx} → {reason}")
    else:
        print("No citation placeholders were inserted.")