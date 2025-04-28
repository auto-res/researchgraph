import os
import json
from typing import Optional
from pydantic import BaseModel
from litellm import completion
from jinja2 import Environment
from langchain_community.document_loaders import PyPDFLoader


class LLMOutput(BaseModel):
    objective: str
    new_method_text: str
    new_method_results: str
    new_method_analysis: str


class WriterSubgraphInputPrep:
    def __init__(self, llm_name: str, output_dir: str):
        self.llm_name = llm_name
        self.output_dir = output_dir
        self.env = Environment()

    def _parse_pdf(self, pdf_path) -> str:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        full_text = "".join(page.page_content.replace("\n", "") for page in pages)
        return full_text

    def _call_llm(
        self, prompt_template: str, full_text: dict, max_retries: int = 3
    ) -> Optional[str]:
        template = self.env.from_string(prompt_template)
        prompt = template.render(full_text)

        for attempt in range(max_retries):
            try:
                response = completion(
                    model=self.llm_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    response_format=LLMOutput,
                )
                structured_output = json.loads(response.choices[0].message.content)
                return structured_output
            except Exception as e:
                print(f"[Attempt {attempt+1}/{max_retries}] Unexpected error: {e}")
        print("Exceeded maximum retries for LLM call.")
        return None

    def _save_summary_to_file(self, summary_data: dict, pdf_path: str):
        base_name = os.path.basename(pdf_path).replace(".pdf", "_summary.txt")
        summary_file_path = os.path.join(self.output_dir, base_name)

        summary_text = f"""
        Objective: {summary_data['objective']}

        New Method Text: {summary_data['new_method_text']}

        New Method Results: {summary_data['new_method_results']}

        New Method Analysis: {summary_data['new_method_analysis']}
        """

        with open(summary_file_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        print(f"Summary saved to: {summary_file_path}")

    def execute(self, pdf_path):
        full_text = self._parse_pdf(pdf_path)

        writer_input_data = self._call_llm(
            writer_subgraph_input_prep_prompt, {"full_text": full_text}
        )
        print(f"writer_input_data: {writer_input_data}")

        if writer_input_data is None:
            raise ValueError("LLM failed to generate structured output.")

        self._save_summary_to_file(writer_input_data, pdf_path)

        return {
            "objective": writer_input_data["objective"],
            "new_method_text": writer_input_data["new_method_text"],
            "new_method_code": "def new_optimizer(): pass",
            "new_method_results": writer_input_data["new_method_results"],
            "new_method_analysis": writer_input_data["new_method_analysis"],
            "github_url": "https://github.com/example/repo",
            "paper_content": {},
            "tex_text": "",
            "github_owner": "mock_owner",
            "repository_name": "mock_repo",
            "branch_name": "mock_branch",
            "add_github_url": "mock_add_url",
            "base_github_url": "mock_base_url",
            "completion": False,
            "devin_url": "mock_devin_url",
        }


writer_subgraph_input_prep_prompt = """
Extract detailed, structured information from the research paper for automated scientific writing.  
**Retain all key details, including equations, datasets, benchmarks, hyperparameters, and comparisons.**  

## **Extract the following as JSON:**
```json
{
    "objective": "<Clearly defined research goal, significance, and hypothesis>",
    "new_method_text": "<Comprehensive methodology, including equations, pseudocode, algorithms, hyperparameters, comparisons, and technical details>",
    "new_method_results": "<Complete experimental results, including datasets, benchmarks, performance metrics, statistical significance tests, and visualizations>",
    "new_method_analysis": "<Findings, theoretical implications, limitations, error analysis, and future research directions>"
}

##**Instructions**:
1. Preserve all extracted information (More information is always better)
- Do not summarize, simplify, or omit any information from the paper. Extract and retain every possible detail.
- new_method_text, new_method_results, and new_method_analysis should each be at least 3000 characters (preferably 4000+), equivalent to at least 1.5 pages of detailed text.
- Objective should be at least 1000 characters, covering the research hypothesis and significance.
2. Methodology must be highly detailed
- Extract full methodology, including:
    - Pseudocode, mathematical derivations, and algorithm explanations.
    - Hyperparameter settings with exact values (learning rate, batch size, optimizer, etc.).
    - Comparison against at least two baseline techniques and explanation of performance differences.
    - Computational complexity and theoretical analysis.
    - Detailed descriptions of how the proposed method works, including intermediate steps and assumptions.
    - All algorithm variations or ablations (e.g., different architectures, hyperparameter tuning, regularization strategies).
3. Experimental results must be comprehensive
- Ensure all relevant details are extracted, including:
    - Dataset descriptions (size, structure, preprocessing techniques).
    - Performance metrics (accuracy, F1-score, loss values, statistical confidence intervals).
    - Ablation studies (effect of hyperparameter changes, component analysis).
    - Exact numerical values (e.g., "Our method achieved 92.3% accuracy, compared to 89.5% for baseline A").
    - Figures, graphs, and tables should be fully described in text, including their exact numerical data.
    - If multiple experiments were conducted, extract all results, even those that seem redundant.
4. Retain discussion, limitations, and future directions
- Provide an in-depth analysis covering:
    - Strengths and weaknesses of the proposed approach.
    - Potential real-world applications and theoretical implications.
    - Clear directions for future research.
    - If different hypotheses or interpretations are mentioned, include all perspectives.

Below is the full text of the research paper:
```
{{ full_text }}
```
"""
