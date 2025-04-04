import json
from pydantic import BaseModel
from jinja2 import Environment
from typing import Optional
from researchgraph.utils.openai_client import openai_client
from logging import getLogger

logger = getLogger(__name__)

env = Environment()


class LLMOutput(BaseModel):
    generated_paper_text: str


class WritingNode:
    def __init__(
        self,
        llm_name: str,
        refine_round: int = 2,
        refine_only: bool = False,
        target_sections: list[str] = [],
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
            "Title": """\n
- Write only the title of the paper in one single line, as plain text with no quotation marks.
    - Example of correct output: Efficient Adaptation of Large Language Models via Low-Rank Optimization
    - Incorrect output: "Efficient Adaptation of Large Language Models via Low-Rank Optimization"
- The title must be concise and descriptive of the paper's concept, but try by creative with it.
- Do not include any explanations, subsections, LaTeX commands (\\title{...}, etc.)""",
            "Abstract": """\n
- Expected length: about 1000 words
- TL;DR of the paper
- What are we trying to do and why is it relevant?
- Why is this hard? 
- How do we solve it (i.e. our contribution!)
- How do we verify that we solved it (e.g. Experiments and results)
- Avoid using itemize, subheadings, or displayed equations in the abstract; keep math in plain text and list contributions inline.
- Please make sure the abstract reads smoothly and is well-motivated. This should be one continuous paragraph with no breaks between the lines.""",
            "Introduction": """\n
- Expected length: about 4000 words (~1–1.5 pages)
- Longer version of the Abstract, i.e. of the entire paper
- What are we trying to do and why is it relevant?
- Why is this hard? 
- How do we solve it (i.e. our contribution!)
- How do we verify that we solved it (e.g. Experiments and results)
- New trend: specifically list your contributions as bullet points
- Extra space? Future work!""",
            "Related Work": """\n
- Expected length: about 3000 words (~1 pages)
- Academic siblings of our work, i.e. alternative attempts in literature at trying to solve the same problem. 
- Goal is to “Compare and contrast” - how does their approach differ in either assumptions or method? If their method is applicable to our Problem Setting I expect a comparison in the experimental section. If not, there needs to be a clear statement why a given method is not applicable. 
- Note: Just describing what another paper is doing is not enough. We need to compare and contrast.""",
            "Background": """\n
- Expected length: about 3000 words (~1 pages)
- Academic Ancestors of our work, i.e. all concepts and prior work that are required for understanding our method. 
- Usually includes a subsection, Problem Setting, which formally introduces the problem setting and notation (Formalism) for our method. Highlights any specific assumptions that are made that are unusual. 
- Note: If our paper introduces a novel problem setting as part of its contributions, it's best to have a separate Section.""",
            "Method": """\n
- Expected length: about 4000 words (~1–1.5 pages)
- What we do. Why we do it. All described using the general Formalism introduced in the Problem Setting and building on top of the concepts / foundations introduced in Background.""",
            "Experimental Setup": """
- Expected length: about 4000 words (~1–1.5 pages)
- How do we test that our stuff works? Introduces a specific instantiation of the Problem Setting and specific implementation details of our Method for this Problem Setting.
- Do not imagine unknown hardware details.
- Includes a description of the dataset, evaluation metrics, important hyperparameters, and implementation details.""",
            "Results": """\n
- Expected length: about 4000 words (~1–1.5 pages)
- Shows the results of running Method on our problem described in Experimental Setup.
- Includes statements on hyperparameters and other potential issues of fairness.
- Only includes results that have actually been run and saved in the logs. Do not hallucinate results that don't exist.
- If results exist: compares to baselines and includes statistics and confidence intervals. 
- If results exist: includes ablation studies to show that specific parts of the method are relevant.
- Discusses limitations of the method.
- Make sure to include all the results from the experiments, and include all relevant figures.""",
            "Conclusions": """\n
- Expected length: about 2000 words (~0.5 pages)
- Do not include \\section{...} or \\subsection{...}.
- Brief recap of the entire paper.
- To keep going with the analogy, you can think of future work as (potential) academic offspring.""",
        }

        self.system_prompt = """\n
A complete LaTeX template is already in place.
Your role is to generate or refine the LaTeX content specifically for the '{{ section }}' section, such as text, equations, figures, tables, and citations—all within the existing document structure.

The final LaTeX content should be returned as the value of the key “generated_paper_text”.

Some tips are provided below:
{{ tips }}

Here is the context of the entire paper:
{{ note }}

**Instructions**:
- Use ONLY the provided information in the context. 
    - **DO NOT add any assumptions, invented data, or details that are not explicitly mentioned in the context.
- **Use ALL information from '{{ note }}'.** However, you are free to organize and structure the content in a natural and logical way, rather than directly following the order or format of `{{ note }}`
    - **It is mandatory to include all details related to methods, experiments, and results.**
    - **Ensure all mathematical equations, pseudocode, experimental setups, configurations, numerical results, and figures/tables are fully incorporated.**
    - When beneficial for clarity, utilize tables or pseudocode to describe mathematical equations, parameter settings, and procedural steps.
    - Avoid overly explanatory or repetitive descriptions that would be self-evident to readers familiar with standard machine learning notation.
- List contributions using \\begin{itemize}...\\end{itemize} in LaTeX. Each item should start with a short title in \\textbf{...} format. Avoid using -, *, or other Markdown bullet styles.
- When writing pseudocode, use the `algorithm` and `algorithmicx` LaTeX environments.
    - Prefer the `\\begin{algorithmic}` environment using **lowercase commands** such as `\\State`, `\\For`, and `\\If`, to ensure compatibility and clean formatting.
- Figures and images are ONLY allowed in the "Results" section. 
    - Use LaTeX float option `[H]` to force placement.  
- All figures must be inserted using the following LaTeX format, using a `width` that reflects the filename:
    ```
    \\includegraphics[width=<appropriate-width>]{images/filename.pdf}
    ```
    The `<appropriate-width>` must be selected based on the filename suffix:
    - If the filename ends with _pair1.pdf or _pair2.pdf, use 0.48\\linewidth and place the figures side by side using subfigure blocks
    - Otherwise (default), use 0.7\\linewidth
- Avoid editor instructions, placeholders, speculative text, or comments like "details are missing."
    - Example: Remove phrases like "Here’s a refined version of the '{{ section }}'," as they are not part of the final document.
    - These phrases are found at the beginning of sections, introducing edits or refinements. Carefully review the start of each section for such instructions and ensure they are eliminated while preserving the actual content.
- **Use \\subsection{...} for any subsections within this section.**
    - Subsection titles should be distinct from the '{{ section }}' title. 
    - **Do not use '\\subsection{ {{ section }} }', or other slight variations. Use more descriptive and unique titles.
    - Avoid creating too many subsections. If the content of a subsection is brief or overlaps significantly with other subsections, merge them to streamline the document. Focus on clarity and brevity over excessive structural division.
- Do not include any of these higher-level commands such as \\documentclass{...}, \\begin{document}, and \\end{document}.
    - Additionally, avoid including section-specific commands such as \\begin{abstract}, \\section{ {{ section }} }, or any other similar environment definitions.
- Be sure to use \\cite or \\citet where relevant, referring to the works provided in the file.
    - **Do not cite anything that is not already in `references.bib`. Do not add any new entries to this.
- Keep the experimental results (figures and tables) only in the Results section, and make sure that any captions are filled in.
- The full paper should be **about 8 pages long**, meaning **each section should contain substantial content**."""

        self.write_prompt_template = """\n
You are tasked with filling in the '{{ section }}' section of a research paper."""

        self.refinement_template = """\n
You are tasked with refining the '{{ section }}' section of a research paper.

Here is the content that needs refinement:
{{ content }}

Pay particular attention to fixing any errors such as:
{{ error_list_prompt }}"""

        self.error_list_prompt = """\n
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
- Incorrect closing of environments, e.g. </end{{figure}}> instead of \\end{{figure}}"""

    def _call_llm(self, prompt: str, system_prompt: str) -> Optional[str]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = openai_client(self.llm_name, message=messages, data_class=LLMOutput)
        if response is not None:
            response = json.loads(response)
            return response["generated_paper_text"]
        else:
            return None

    def _write(self, note: str, section_name: str) -> str:
        prompt = self._generate_write_prompt(section_name, note)
        system_prompt = self._render_system_prompt(section_name, note)
        content = self._call_llm(prompt=prompt, system_prompt=system_prompt)
        if content is None:
            raise RuntimeError(
                f"Failed to generate content for section: {section_name}. The LLM returned None."
            )
        return content

    def _refine(self, note: str, section_name: str, content: str) -> str:
        for round_num in range(self.refine_round):
            prompt = self._generate_refinement_prompt(section_name, note, content)
            system_prompt = self._render_system_prompt(section_name, note)
            refine_content = self._call_llm(prompt=prompt, system_prompt=system_prompt)
            if refine_content is None:
                logger.info(
                    f"Refinement failed for {section_name} at round {round_num + 1}. Keeping previous content."
                )
                break
            content = refine_content
        return content

    def _relate_work(
        self, content: str
    ) -> str:  # TODO: Implement functionality to manage retrieved papers in a centralized references file (e.g., references.bib).
        return content  #       Generate descriptions based on information in RelatedWorkNode.

    def _render_system_prompt(self, section: str, note: str) -> str:
        template = env.from_string(self.system_prompt)
        return template.render(
            section=section,
            note=note,
            tips=self.per_section_tips_prompt_dict.get(section, ""),
        )

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

    def execute(self, note: str) -> dict:
        paper_content = {}
        for section in self.target_sections:
            logger.info(f"Writing {section}")
            if not self.refine_only:
                # generate and refine
                initial_content = self._write(note, section)
                # initial_content = self._relate_work(initial_content)
                refined_content = self._refine(note, section, initial_content)
            else:
                # refine only
                initial_content = paper_content.get(section, "")
                # initial_content = getattr(state, section)
                refined_content = self._refine(note, section, initial_content)

            paper_content[section] = refined_content
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
