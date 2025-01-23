from jinja2 import Environment

env = Environment()

per_section_tips = {
"Title": """
- Write only the title of the paper, in **one single line**.
- The title must be concise and descriptive of the paper's concept, but try by creative with it.
- Do not include any explanations, descriptions, comments, or additional sentences—**strictly output only the title**.
""", 
    "Abstract": """
- TL;DR of the paper
- What are we trying to do and why is it relevant?
- Why is this hard? 
- How do we solve it (i.e. our contribution!)
- How do we verify that we solved it (e.g. Experiments and results)

Please make sure the abstract reads smoothly and is well-motivated. This should be one continuous paragraph with no breaks between the lines.
""",
    "Introduction": """
- Longer version of the Abstract, i.e. of the entire paper
- What are we trying to do and why is it relevant?
- Why is this hard? 
- How do we solve it (i.e. our contribution!)
- How do we verify that we solved it (e.g. Experiments and results)
- New trend: specifically list your contributions as bullet points
- Extra space? Future work!
""",
    "Related Work": """
- Academic siblings of our work, i.e. alternative attempts in literature at trying to solve the same problem. 
- Goal is to “Compare and contrast” - how does their approach differ in either assumptions or method? If their method is applicable to our Problem Setting I expect a comparison in the experimental section. If not, there needs to be a clear statement why a given method is not applicable. 
- Note: Just describing what another paper is doing is not enough. We need to compare and contrast.
""",
    "Background": """
- Academic Ancestors of our work, i.e. all concepts and prior work that are required for understanding our method. 
- Usually includes a subsection, Problem Setting, which formally introduces the problem setting and notation (Formalism) for our method. Highlights any specific assumptions that are made that are unusual. 
- Note: If our paper introduces a novel problem setting as part of its contributions, it's best to have a separate Section.
""",
    "Method": """
- What we do. Why we do it. All described using the general Formalism introduced in the Problem Setting and building on top of the concepts / foundations introduced in Background.
""",
    "Experimental Setup": """
- How do we test that our stuff works? Introduces a specific instantiation of the Problem Setting and specific implementation details of our Method for this Problem Setting.
- Do not imagine unknown hardware details.
- Includes a description of the dataset, evaluation metrics, important hyperparameters, and implementation details.
""",
    "Results": """
- Shows the results of running Method on our problem described in Experimental Setup.
- Includes statements on hyperparameters and other potential issues of fairness.
- Only includes results that have actually been run and saved in the logs. Do not hallucinate results that don't exist.
- If results exist: compares to baselines and includes statistics and confidence intervals. 
- If results exist: includes ablation studies to show that specific parts of the method are relevant.
- Discusses limitations of the method.
- Make sure to include all the results from the experiments, and include all relevant figures.
""",
    "Conclusions": """
- Brief recap of the entire paper.
- To keep going with the analogy, you can think of future work as (potential) academic offspring.
""",
}

error_list = """
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

write_template = """
You are tasked with filling in the '{{ section }}' section of a research paper.
Your response MUST be a valid JSON object with a single key, "paper_text".
The value of "paper_text" should be the content of the '{{ section }}' section in LaTeX format.

Example JSON output:
```json
{
  "paper_text": "Content of the '{{ section }}' section, formatted in LaTeX, including subsections as needed."
}
```

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

def generate_write_prompt(section: str, note: str) -> str:
    """
    Generate a write prompt for a specific section using Jinja2.
    """
    template = env.from_string(write_template)
    return template.render(section=section, note=note, tips=per_section_tips.get(section, ""))

# Generate refinement prompt template
refinement_template = """
Great job! Now criticize and refine only the {{ section }} that you just wrote. Return the refined content as a JSON object with the key "paper_text".

Your response MUST be a valid JSON object with a single key, "paper_text".
The value of "paper_text" should be the refined content of the '{{ section }}' section in LaTeX format.

Here is the content that needs refinement:
{{ content }}

Example JSON output:
```json
{
  "paper_text": "Refined content of the '{{ section }}' section, formatted in LaTeX, including subsections as needed."
}
```

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
 {{ error_list }}

"""

def generate_refinement_prompt(section: str, note: str, content: str) -> str:
    """
    Generate a refinement prompt for a specific section using Jinja2.
    """
    template = env.from_string(refinement_template)
    return template.render(
        section=section, 
        note=note,
        content=content,
        tips=per_section_tips.get(section, ""), 
        error_list=error_list
    )

related_work_prompt = f"""Please fill in the Related Work of the writeup. Some tips are provided below:

{per_section_tips["Related Work"]}

For this section, very briefly sketch out the structure of the section, and clearly indicate what papers you intend to include.
Do this all in LaTeX comments using %.
The related work should be concise, only plan to discuss the most relevant work.
Do not modify `references.bib` to add any new citations, this will be filled in at a later stage.

Be sure to first name the file and use *SEARCH/REPLACE* blocks to perform these edits.
"""

citation_system_msg = """You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.
You have already written an initial draft of the paper and now you are looking to add missing citations to related papers throughout the paper.
The related work section already has some initial comments on which papers to add and discuss.

Focus on completing the existing write-up and do not add entirely new elements unless necessary.
Ensure every point in the paper is substantiated with sufficient evidence.
Feel free to add more cites to a particular point if there is only one or two references.
Ensure no paper is cited without a corresponding reference in the `references.bib` file.
Ensure each paragraph of the related work has sufficient background, e.g. a few papers cited.
You will be given access to the Semantic Scholar API, only add citations that you have found using the API.
Aim to discuss a broad range of relevant papers, not just the most popular ones.
Make sure not to copy verbatim from prior literature to avoid plagiarism.

You will be prompted to give a precise description of where and how to add the cite, and a search query for the paper to be cited.
Finally, you will select the most relevant cite from the search results (top 10 results will be shown).
You will have {total_rounds} rounds to add to the references, but do not need to use them all.

DO NOT ADD A CITATION THAT ALREADY EXISTS!"""

citation_first_prompt = '''Round {current_round}/{total_rounds}:

You have written this LaTeX draft so far:

"""
{draft}
"""

Identify the most important citation that you still need to add, and the query to find the paper.

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason over the paper and identify where citations should be added.
If no more citations are needed, add "No more citations needed" to your thoughts.
Do not add "No more citations needed" if you are adding citations this round.

In <JSON>, respond in JSON format with the following fields:
- "Description": A precise description of the required edit, along with the proposed text and location where it should be made.
- "Query": The search query to find the paper (e.g. attention is all you need).

Ensure the description is sufficient to make the change without further context. Someone else will make the change.
The query will work best if you are able to recall the exact name of the paper you are looking for, or the authors.
This JSON will be automatically parsed, so ensure the format is precise.'''

citation_second_prompt = """Search has recovered the following articles:

{papers}

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason over the search results and identify which citation best fits your paper and the location is to be added at.
If none are appropriate, add "Do not add any" to your thoughts.

In <JSON>, respond in JSON format with the following fields:
- "Selected": A list of the indices of the selected papers to be cited, e.g. "[0, 1]". Can be "[]" if no papers are selected. This must be a string.
- "Description": Update the previous description of the required edit if needed. Ensure that any cites precisely match the name in the bibtex!!!

Do not select papers that are already in the `references.bib` file at the top of the draft, or if the same citation exists under a different name.
This JSON will be automatically parsed, so ensure the format is precise."""