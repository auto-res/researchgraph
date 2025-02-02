import re
import json
from pydantic import BaseModel
from jinja2 import Template, Environment
from litellm import completion
# from researchgraph.nodes.writingnode.prompts.writeup_prompt import (
#     generate_write_prompt,
#     generate_refinement_prompt,
# )

env = Environment()

# NOTE: These regex rules are used to classify the keys in state[""] into specific sections.
regex_rules = {
    "Title": r"^(title|objective)$",
    "Methods": r".*_method_text$",
    "Codes": r".*_method_code$",
    "Results": r".*_results$",
    "Analyses": r".*_analysis$",
    # "Related Work": r"^(arxiv_url|github_url)$"
}


class LLMOutput(BaseModel):
    generated_paper_text: str


class WriteupNode:
    def __init__(
        self,
        llm_name: str,
        refine_round: int = 2,
        refine_only: bool = False,
        target_sections: list[str] = None,
    ):
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
        self.per_section_tips_prompt_dict = {
            "Title": """
- Write only the title of the paper, in **one single line**.
- The title must be concise and descriptive of the paper's concept, but try by creative with it.
- Do not include any explanations, descriptions, comments, or additional sentences—**strictly output only the title**.""",
            "Abstract": """
- TL;DR of the paper
- What are we trying to do and why is it relevant?
- Why is this hard? 
- How do we solve it (i.e. our contribution!)
- How do we verify that we solved it (e.g. Experiments and results)
Please make sure the abstract reads smoothly and is well-motivated. This should be one continuous paragraph with no breaks between the lines.""",
            "Introduction": """
- Longer version of the Abstract, i.e. of the entire paper
- What are we trying to do and why is it relevant?
- Why is this hard? 
- How do we solve it (i.e. our contribution!)
- How do we verify that we solved it (e.g. Experiments and results)
- New trend: specifically list your contributions as bullet points
- Extra space? Future work!""",
            "Related Work": """
- Academic siblings of our work, i.e. alternative attempts in literature at trying to solve the same problem. 
- Goal is to “Compare and contrast” - how does their approach differ in either assumptions or method? If their method is applicable to our Problem Setting I expect a comparison in the experimental section. If not, there needs to be a clear statement why a given method is not applicable. 
- Note: Just describing what another paper is doing is not enough. We need to compare and contrast.""",
            "Background": """
- Academic Ancestors of our work, i.e. all concepts and prior work that are required for understanding our method. 
- Usually includes a subsection, Problem Setting, which formally introduces the problem setting and notation (Formalism) for our method. Highlights any specific assumptions that are made that are unusual. 
- Note: If our paper introduces a novel problem setting as part of its contributions, it's best to have a separate Section.""",
            "Method": """
- What we do. Why we do it. All described using the general Formalism introduced in the Problem Setting and building on top of the concepts / foundations introduced in Background.""",
            "Experimental Setup": """
- How do we test that our stuff works? Introduces a specific instantiation of the Problem Setting and specific implementation details of our Method for this Problem Setting.
- Do not imagine unknown hardware details.
- Includes a description of the dataset, evaluation metrics, important hyperparameters, and implementation details.""",
            "Results": """
- Shows the results of running Method on our problem described in Experimental Setup.
- Includes statements on hyperparameters and other potential issues of fairness.
- Only includes results that have actually been run and saved in the logs. Do not hallucinate results that don't exist.
- If results exist: compares to baselines and includes statistics and confidence intervals. 
- If results exist: includes ablation studies to show that specific parts of the method are relevant.
- Discusses limitations of the method.
- Make sure to include all the results from the experiments, and include all relevant figures.""",
            "Conclusions": """
- Brief recap of the entire paper.
- To keep going with the analogy, you can think of future work as (potential) academic offspring.""",
        }

        self.write_prompt_template = """
You are tasked with filling in the '{{ section }}' section of a research paper.
The generated text should be output as a value for the key “generated_papet_text”.
The value of "generated_paper_text" should be the content of the '{{ section }}' section in LaTeX format.
Some tips are provided below:
{{ tips }}
Here is the context of the entire paper:
{{ note }}
**Instructions**:
- Use ONLY the provided information in the context. 
    -**DO NOT add any assumptions, invented data, or details that are not explicitly mentioned in the context.
- Avoid placeholders, speculative text, or comments like "details are missing."
- Do not include section headings such as \section{...}. 
    -**Use \subsection{Subsection Title} for any subsections within this section.** The titles of the subsections should be derived from the content you are generating. The "Title" section should not have any subsections.
    - Ensure that identical or similar {{ section }} sections are completely removed. Subsection names should be derived from the content of the section.
- Be sure to use \cite or \citet where relevant, referring to the works provided in the file.
    -**Do not cite anything that is not already in `references.bib`. Do not add any new entries to this.
- Keep the experimental results (figures and tables) only in the Results section, and make sure that any captions are filled in.
- Ensure that editor instructions are completely removed:
  - Example: Remove phrases like "Here’s a refined version of the Background section," as they are not part of the final document.
  - These phrases are found at the beginning of sections, introducing edits or refinements. Carefully review the start of each section for such instructions and ensure they are eliminated while preserving the actual content.
Before every paragraph, please include a brief description of what you plan to write in that paragraph in a LaTeX comment (% ...).
Be sure to first name the file and use *SEARCH/REPLACE* blocks to perform these edits.

"""

        self.refinement_template = """
Great job! Now criticize and refine only the {{ section }} that you just wrote. Return the refined content as a JSON object with the key "paper_text".
The generated text should be output as a value for the key “generated_papet_text”.
The value of "generated_paper_text" should be the refined content of the '{{ section }}' section in LaTeX format.
Here is the content that needs refinement:
{{ content }}
Some tips are provided below:
{{ tips }}
Here is the context of the entire paper:
{{ note }}
**Instructions**:
- Use ONLY the provided information in the context. 
    -**DO NOT add any assumptions, invented data, or details that are not explicitly mentioned in the context.
- Avoid placeholders, speculative text, or comments like "details are missing."
- Do not include section headings such as \section{...}. 
    -**Convert all headings into subsections using \subsection{...}, except for the "Title" section. 
    - Ensure that identical or similar sections are completely removed. The titles of the subsections should be derived from the content you are generating.
    - Avoid creating too many subsections. If the content of a subsection is brief or overlaps significantly with other subsections, merge them to streamline the document. Focus on clarity and brevity over excessive structural division.
- Identify any redundancies (e.g. repeated figures or repeated text), if there are any, decide where in the paper things should be cut.
- Identify where we can save space, and be more concise without weakening the message of the text.
- Ensure that editor instructions are completely removed:
  - Example: Remove phrases like "Here’s a refined version of the Background section," as they are not part of the final document.
  - These phrases are found at the beginning of sections, introducing edits or refinements. Carefully review the start of each section for such instructions and ensure they are eliminated while preserving the actual content.
Pay particular attention to fixing any errors such as:
{{ error_list_prompt }}"""

        self.error_list_prompt = """
- Unenclosed math symbols
- Only reference figures that exist in our directory
- LaTeX syntax errors
- Numerical results that do not come from explicit experiments and logs
- Repeatedly defined figure labels
- References to papers that are not in the .bib file, DO NOT ADD ANY NEW CITATIONS!
- Unnecessary verbosity or repetition, unclear text
- Results or insights in the {{ note }} that have not yet need included
- Any relevant figures that have not yet been included in the text
- Closing any \\begin{{figure}} with a \\end{{figure}} and \\begin{{table}} with a \\end{{table}}, etc.
- Duplicate headers, e.g. duplicated \\section{{Introduction}} or \\end{{document}}
- Unescaped symbols, e.g. shakespeare_char should be shakespeare\\_char in text
- Incorrect closing of environments, e.g. </end{{figure}}> instead of \\end{{figure}}
"""

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
            for key, value in state.items():
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
            response_format=LLMOutput,
        )
        structured_output = json.loads(response.choices[0].message.content)
        return structured_output["generated_paper_text"]

    def _write(self, note: str, section_name: str) -> str:
        prompt = self._generate_write_prompt(section_name, note)
        content = self._call_llm(prompt)
        return content

    def _refine(self, note: str, section_name: str, content: str) -> str:
        for _ in range(self.refine_round):
            prompt = self._generate_refinement_prompt(section_name, note, content)
            content = self._call_llm(prompt)
        return content

    def _relate_work(
        self, content: str
    ) -> str:  # TODO: Implement functionality to manage retrieved papers in a centralized references file (e.g., references.bib).
        return content  #       Generate descriptions based on information in RelatedWorkNode.

    def _generate_write_prompt(self, section: str, note: str) -> str:
        """
        Generate a write prompt for a specific section using Jinja2.
        """
        template = env.from_string(self.write_prompt_template)
        return template.render(
            section=section,
            note=note,
            tips=self.per_section_tips_prompt_dict.get(section, ""),
        )

    def _generate_refinement_prompt(self, section: str, note: str, content: str) -> str:
        """
        Generate a refinement prompt for a specific section using Jinja2.
        """
        template = env.from_string(self.refinement_template)
        return template.render(
            section=section,
            note=note,
            content=content,
            tips=self.per_section_tips_prompt_dict.get(section, ""),
            error_list_prompt=self.error_list_prompt,
        )

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
        note = self._generate_note(state)
        paper_content = {}
        for section in self.target_sections:
            print(f"section: {section}")
            if not self.refine_only:
                # generate and refine
                initial_content = self._write(note, section)
                # initial_content = self._relate_work(initial_content)
                cleaned_initial_content = self._clean_meta_information(initial_content)
                refined_content = self._refine(note, section, cleaned_initial_content)
                # refined_content = self._refine(note, section, initial_content)
            else:
                # refine only
                # initial_content = paper_content.get(section, "")
                initial_content = getattr(state, section)
                refined_content = self._refine(note, section, initial_content)

            final_content = self._clean_meta_information(refined_content)
            paper_content[section] = final_content
        return paper_content


# related_work_prompt = f"""Please fill in the Related Work of the writeup. Some tips are provided below:
# {per_section_tips["Related Work"]}
# For this section, very briefly sketch out the structure of the section, and clearly indicate what papers you intend to include.
# Do this all in LaTeX comments using %.
# The related work should be concise, only plan to discuss the most relevant work.
# Do not modify `references.bib` to add any new citations, this will be filled in at a later stage.
# Be sure to first name the file and use *SEARCH/REPLACE* blocks to perform these edits."""

# citation_system_msg = """You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.
# You have already written an initial draft of the paper and now you are looking to add missing citations to related papers throughout the paper.
# The related work section already has some initial comments on which papers to add and discuss.
# Focus on completing the existing write-up and do not add entirely new elements unless necessary.
# Ensure every point in the paper is substantiated with sufficient evidence.
# Feel free to add more cites to a particular point if there is only one or two references.
# Ensure no paper is cited without a corresponding reference in the `references.bib` file.
# Ensure each paragraph of the related work has sufficient background, e.g. a few papers cited.
# You will be given access to the Semantic Scholar API, only add citations that you have found using the API.
# Aim to discuss a broad range of relevant papers, not just the most popular ones.
# Make sure not to copy verbatim from prior literature to avoid plagiarism.
# You will be prompted to give a precise description of where and how to add the cite, and a search query for the paper to be cited.
# Finally, you will select the most relevant cite from the search results (top 10 results will be shown).
# You will have {total_rounds} rounds to add to the references, but do not need to use them all.
# DO NOT ADD A CITATION THAT ALREADY EXISTS!"""

# citation_first_prompt = '''Round {current_round}/{total_rounds}:
# You have written this LaTeX draft so far:
# """
# {draft}
# """
# Identify the most important citation that you still need to add, and the query to find the paper.
# Respond in the following format:
# THOUGHT:
# <THOUGHT>
# RESPONSE:
# ```json
# <JSON>
# ```
# In <THOUGHT>, first briefly reason over the paper and identify where citations should be added.
# If no more citations are needed, add "No more citations needed" to your thoughts.
# Do not add "No more citations needed" if you are adding citations this round.
# In <JSON>, respond in JSON format with the following fields:
# - "Description": A precise description of the required edit, along with the proposed text and location where it should be made.
# - "Query": The search query to find the paper (e.g. attention is all you need).
# Ensure the description is sufficient to make the change without further context. Someone else will make the change.
# The query will work best if you are able to recall the exact name of the paper you are looking for, or the authors.
# This JSON will be automatically parsed, so ensure the format is precise.'''

# citation_second_prompt = """Search has recovered the following articles:
# {papers}
# Respond in the following format:
# THOUGHT:
# <THOUGHT>
# RESPONSE:
# ```json
# <JSON>
# ```
# In <THOUGHT>, first briefly reason over the search results and identify which citation best fits your paper and the location is to be added at.
# If none are appropriate, add "Do not add any" to your thoughts.
# In <JSON>, respond in JSON format with the following fields:
# - "Selected": A list of the indices of the selected papers to be cited, e.g. "[0, 1]". Can be "[]" if no papers are selected. This must be a string.
# - "Description": Update the previous description of the required edit if needed. Ensure that any cites precisely match the name in the bibtex!!!
# Do not select papers that are already in the `references.bib` file at the top of the draft, or if the same citation exists under a different name.
# This JSON will be automatically parsed, so ensure the format is precise."""

if __name__ == "__main__":
    state = {
        "objective": "Researching optimizers for fine-tuning LLMs.",
        "base_method_text": "Baseline method description...",
        "add_method_text": "Added method description...",
        "new_method_text": ["New combined method description..."],
        "base_method_code": "def base_method(): pass",
        "add_method_code": "def add_method(): pass",
        "new_method_code": ["def new_method(): pass"],
        "base_method_results": "Accuracy: 0.85",
        "add_method_results": "Accuracy: 0.88",
        "new_method_results": ["Accuracy: 0.92"],
    }
    llm_name = "gpt-4o-2024-11-20"
    refine_round = 2
    paper_content = WriteupNode(
        llm_name=llm_name,
        refine_round=refine_round,
    ).execute(state)
    print(paper_content)
