import re
import json
from jinja2 import Template
from researchgraph.core.node import Node
from litellm import completion
from pydantic import BaseModel
from researchgraph.nodes.writingnode.prompts.writeup_prompt import (
    generate_write_prompt,
    generate_refinement_prompt,
)

# NOTE: These regex rules are used to classify the keys in state[""] into specific sections.
regex_rules = {
    "Title": r"^(title|objective)$",
    "Methods": r".*_method_text$",
    "Codes": r".*_method_code$",
    "Results": r".*_results$",
    "Analyses": r".*_analysis$",
    # "Related Work": r"^(arxiv_url|github_url)$"
}


class WriteupResponse(BaseModel):
    paper_text: str


class WriteupNode(Node):
    def __init__(
        self,
        input_key: list[str],
        output_key: list[str],
        llm_name: str,
        refine_round: int = 2,
        refine_only: bool = False,
        target_sections: list[str] = None,
    ):
        super().__init__(input_key, output_key)
        self.llm_name = llm_name
        self.refine_round = refine_round
        self.refine_only = refine_only
        self.target_sections = target_sections or [
            "Title",
            "Abstract",
            "Introduction",
            "Related work",
            "Background",
            "Method",
            "Experimental setup",
            "Results",
            "Conclusions",
        ]
        # self.dynamicmodel = self._create_dynamic_model(DynamicModel)

    def _generate_note(self, state: dict) -> str:
        template = Template("""
        {% for section, items in sections.items() %}
        # {{ section }}
        {% for key, value in items.items() %}
        {{ key }}: {{ value }}
        {% endfor %}
        {% endfor %}
        """)

        sections = {}
        for section, pattern in regex_rules.items():
            matched_items = {}
            for key, value in state.dict().items():
                if re.search(pattern, key):
                    matched_items[key] = (
                        ", ".join(value) if isinstance(value, list) else value
                    )
            sections[section] = matched_items
        # print(f"note: {template.render(sections=sections)}")
        return template.render(sections=sections)

    def _call_llm(self, prompt: str) -> str:
        response = completion(
            model=self.llm_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            response_format=WriteupResponse,
        )
        structured_output = json.loads(response.choices[0].message.content)
        return structured_output["paper_text"]

    def _write(self, note: str, section_name: str) -> str:
        prompt = generate_write_prompt(section_name, note)
        content = self._call_llm(prompt)
        return content

    def _refine(self, note: str, section_name: str, content: str) -> str:
        for _ in range(self.refine_round):
            prompt = generate_refinement_prompt(section_name, note, content)
            content = self._call_llm(prompt)
        return content

    def _relate_work(
        self, content: str
    ) -> str:  # TODO: Implement functionality to manage retrieved papers in a centralized references file (e.g., references.bib).
        return content  #       Generate descriptions based on information in RelatedWorkNode.

    def _clean_meta_information(
        self, text: str
    ) -> str:  # TODO: Combine with prompts to more accurately remove unnecessary meta-information and artifacts.
        meta_patterns = [
            r"Here(?: is|'s) \w+ version.*?(\.|\n)",
            r"This section discusses.*?(\.|\n)",
            r"Before every paragraph.*?(\.|\n)",
            r"Refinement Pass \d+.*?(\.|\n)",
            r"Do not include.*?(\.|\n)",
            r"Certainly!*?(\.|\n)",
            r"_.*?@cref",
            r"^\s*[-*]\s+.*?$",  # Bullet point instructions (e.g., "- Be concise.")
            r"^(```|''')[\w]*\n",  # Opening code block markers (e.g., ```latex)
            r"(```|''')\s*$",  # Closing code block markers
            r"\[\s*[a-zA-Z\s]+\s*\]",  # Unresolved placeholders (e.g., [specific reason])
            r"^%\s*.*$",  # Lines starting with '%'
            r"---\n",
        ]
        combined_pattern = "|".join(meta_patterns)
        text = re.sub(
            combined_pattern, "", text, flags=re.DOTALL | re.IGNORECASE | re.MULTILINE
        )

        text = re.sub(
            r"^#{1,}\s*", "", text, flags=re.MULTILINE
        )  # Remove '###' but keep the title content
        text = re.sub(
            r"([_%&#$])", r"\\\1", text
        )  # Escape LaTeX special characters: _, %, &, #, $
        text = re.sub(
            r"\*\*(.*?)\*\*", r"\\textbf{\1}", text
        )  # Convert Markdown bold (**text**) to LaTeX bold (\textbf{text})
        text = re.sub(
            r"(\n\s*){2,}", "\n\n", text.strip()
        )  # Replace consecutive blank lines (2 or more) with a single blank line
        cleaned_text = "\n".join(line.strip() for line in text.splitlines())
        return cleaned_text

    def execute(self, state: dict) -> dict:
        # paper_content = state.get(self.output_key[0], {})
        paper_content = getattr(state, self.output_key[0])
        note = self._generate_note(state)

        for section in self.target_sections:
            print(f"section: {section}")
            if not self.refine_only:
                # generate and refine
                initial_content = self._write(note, section)
                # initial_content = self._relate_work(initial_content)
                cleaned_initial_content = self._clean_meta_information(initial_content)
                refined_content = self._refine(note, section, cleaned_initial_content)
            else:
                # refine only
                # initial_content = paper_content.get(section, "")
                initial_content = getattr(state, section)
                refined_content = self._refine(note, section, initial_content)

            final_content = self._clean_meta_information(refined_content)
            paper_content[section] = final_content

        print(f"paper_content: {paper_content}")
        return {self.output_key[0]: paper_content}
