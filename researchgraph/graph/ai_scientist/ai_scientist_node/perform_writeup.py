import os
import os.path as osp
import json
from typing import Any, Optional
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
from dataclasses import dataclass
from researchgraph.graph.ai_scientist.ai_scientist_node.generate_ideas import search_for_papers
from researchgraph.graph.ai_scientist.ai_scientist_node.llm import get_response_from_llm, extract_json_between_markers
from researchgraph.writingnode.texnode import LatexNode


per_section_tips = {
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
    "Conclusion": """
- Brief recap of the entire paper.
- To keep going with the analogy, you can think of future work as (potential) academic offspring.
""",
}

error_list = """- Unenclosed math symbols
- Only reference figures that exist in our directory
- LaTeX syntax errors
- Numerical results that do not come from explicit experiments and logs
- Repeatedly defined figure labels
- References to papers that are not in the .bib file, DO NOT ADD ANY NEW CITATIONS!
- Unnecessary verbosity or repetition, unclear text
- Results or insights in the `notes.txt` that have not yet need included
- Any relevant figures that have not yet been included in the text
- Closing any \\begin{{figure}} with a \\end{{figure}} and \\begin{{table}} with a \\end{{table}}, etc.
- Duplicate headers, e.g. duplicated \\section{{Introduction}} or \\end{{document}}
- Unescaped symbols, e.g. shakespeare_char should be shakespeare\\_char in text
- Incorrect closing of environments, e.g. </end{{figure}}> instead of \\end{{figure}}
"""

refinement_prompt = (
    """Great job! Now criticize and refine only the {section} that you just wrote.
Make this complete in this pass, do not leave any placeholders.

Pay particular attention to fixing any errors such as:
"""
    + error_list
)

second_refinement_prompt = (
    """Criticize and refine the {section} only. Recall the advice:
{tips}
Make this complete in this pass, do not leave any placeholders.

Pay attention to how it fits in with the rest of the paper.
Identify any redundancies (e.g. repeated figures or repeated text), if there are any, decide where in the paper things should be cut.
Identify where we can save space, and be more concise without weakening the message of the text.
Fix any remaining errors as before:
"""
    + error_list
)

# CITATION HELPERS
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


@dataclass
class CitationContext:
    client: Any
    model: str
    draft: str
    current_round: int
    total_rounds: int


class ComponentBase:
    def __init__(self, writeup_file: str, exp_file: str, notes: str, model: str, io: InputOutput):
        self.main_model = self.select_model(model)
        self.coder = Coder.create(
            main_model=self.main_model,
            fnames=[exp_file, writeup_file, notes],
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )

    def select_model(self, model: str | Model) -> Model:
        model_mapping = {
            "deepseek-coder-v2-0724": "deepseek/deepseek-coder",
            "llama3.1-405b": "openrouter/meta-llama/llama-3.1-405b-instruct"
        }

        if isinstance(model, str):
            model_name = model_mapping.get(model)
            if model_name is None:
                raise ValueError(f"Unknown model identifier: {model}")
            return Model(model_name)
        
        return model


class BaseSection:
    def __init__(self, coder: Coder, section_name: str, section_key: str):
        self.coder = coder
        self.section_name = section_name
        self.section_key = section_key

    def generate_prompt(self, prompt_type: str) -> str:
        prompt_templates = {
            "write": f"""Please fill in the {self.section_name} of the writeup. Some tips are provided below:
            {per_section_tips[self.section_key]}

            Be sure to use \cite or \citet where relevant, referring to the works provided in the file.
            Do not cite anything that is not already in `references.bib`. Do not add any new entries to this.

            Keep the experimental results (figures and tables) only in the Results section, and make sure that any captions are filled in.
            In this pass, do not reference anything in later sections of the paper.

            Before every paragraph, please include a brief description of what you plan to write in that paragraph in a comment.

            Be sure to first name the file and use *SEARCH/REPLACE* blocks to perform these edits.
            """, 
            "refine": refinement_prompt.format(section=self.section_key), 
            "second_refine": second_refinement_prompt.format(section=self.section_key, tips=per_section_tips[self.section_key]), 
        }
        
        if prompt_type not in prompt_templates:
            raise ValueError("Invalid prompt type specified.")
        
        prompt = prompt_templates[prompt_type]
        return prompt.replace(r"{{", "{").replace(r"}}", "}")

    def write(self):
        prompt = self.generate_prompt("write")
        self.coder.run(prompt)

    def refine(self):
        prompt = self.generate_prompt("refine")
        self.coder.run(prompt)

    def second_refine(self) -> Any:
        prompt = self.generate_prompt("second_refine")
        return self.coder.run(prompt)
        

class RelatedWorkSection(BaseSection):
    def __init__(self, coder: Coder):
        super().__init__(coder, "Related Work", "Related Work")

    def generate_prompt(self, prompt_type) -> str:
        if prompt_type == "write":
            return f"""Please fill in the Related Work of the writeup. Some tips are provided below:

            {per_section_tips["Related Work"]}

            For this section, very briefly sketch out the structure of the section, and clearly indicate what papers you intend to include.
            Do this all in LaTeX comments using %.
            The related work should be concise, only plan to discuss the most relevant work.
            Do not modify `references.bib` to add any new citations, this will be filled in at a later stage.

            Be sure to first name the file and use *SEARCH/REPLACE* blocks to perform these edits.
            """
        return super().generate_prompt(prompt_type)


class WriteupComponent(ComponentBase):
    def __init__(
        self, 
        writeup_file: str, 
        exp_file: str, 
        notes: str, 
        model: str | Model, 
        io: InputOutput
    ):
        super().__init__(writeup_file, exp_file, notes, model, io)

        self.sections = {
            "Abstract": BaseSection(self.coder, "Title and Abstract", "Abstract"),
            "Introduction": BaseSection(self.coder, "Introduction", "Introduction"),
            "Background": BaseSection(self.coder, "Background", "Background"),
            "Method": BaseSection(self.coder, "Method", "Method"),
            "Experimental Setup": BaseSection(self.coder, "Experimental Setup", "Experimental Setup"),
            "Results": BaseSection(self.coder, "Results", "Results"),
            "Conclusion": BaseSection(self.coder, "Conclusion", "Conclusion"),
            "Related Work": RelatedWorkSection(self.coder),
        }

    def perform_writeup(self):
        for section_name, section in self.sections.items():
            try:
                section.write()
            except Exception as e:
                print(f"Error in section '{section_name}': {e}")

    def perform_refinement(self):
        for section in self.sections.values():
            section.refine()

    def perform_second_refinement(self) -> dict[str, Any]:
        results = {}
        for section_name, section in self.sections.items():
            results[section_name] = section.second_refine()
        return results

    # PERFORM WRITEUP
    def __call__(
        self,
        idea: dict[str, Any],
        folder_name: str,
        memory_: dict[str, Any],
        cite_client: Any,
        cite_model: str,
        num_cite_rounds=20,
    ) -> dict[str, Any]:
        
        memory_["writeup"] = False

        self.perform_writeup()
        self.perform_refinement()
        second_refinement_results = self.perform_second_refinement()

        memory_["writeup_content"] = second_refinement_results
        memory_["is_writeup_successful"] = True

        self.add_citations(folder_name, cite_client, cite_model, num_cite_rounds)

        # Generate PDF after citations have been added
        tex_node = LatexNode(memory_["writeup_content"])
        tex_node.setup_latex_utils(self.coder)  # Use the shared Coder instance
        # // TODO: the path must be changed correctly
        template_folder = "/workspaces/researchgraph/researchgraph/graph/ai_scientist/templates/2d_diffusion"
        figures_folder = "/workspaces/researchgraph/researchgraph/graph/ai_scientist/"
        pdf_file = "/workspaces/researchgraph/data/test.pdf"

        tex_node.generate_latex(template_folder, figures_folder, pdf_file)

        return memory_

    def add_citations(self, folder_name: str, cite_client: Any, cite_model: str, num_cite_rounds: int):
        for round_num in range(num_cite_rounds):
            file_path = osp.join(folder_name, "latex", "template.tex")
            with open(file_path, "r") as f:
                draft = f.read()

            context = CitationContext(
                client=cite_client,
                model=cite_model,
                draft=draft,
                current_round=round_num,
                total_rounds=num_cite_rounds
            )
            prompt, done = self.get_citation_aider_prompt(context, msg_history=None)
            if done:
                break
            
            if prompt is not None:
                # extract bibtex string
                bibtex_string = prompt.split('"""')[1]
                # insert this into draft before the "\end{filecontents}" line
                search_str = r"\end{filecontents}"
                draft = draft.replace(search_str, f"{bibtex_string}{search_str}")

                with open(file_path, "w") as f:
                    f.write(draft)

                self.coder_out = self.coder.run(prompt)

    def get_citation_aider_prompt(self, context: CitationContext, msg_history: Optional[list]) -> tuple[Optional[str], bool]:
        if msg_history is None:
            msg_history = []
        try:
            # Initial LLM request to determine if more citations are needed
            text, msg_history = self._get_initial_llm_response(context, msg_history)

            if "No more citations needed" in text:
                print("No more citations needed.")
                return None, True

            ## Parse the output and extract JSON
            json_output = self._extract_json_output(text)
            query = json_output["Query"]
            papers = search_for_papers(query)
        except Exception as e:
            print(f"Error during initial citation retrieval: {e}")
            return None, False

        if papers is None:
            print("No papers found.")
            return None, False

        # Format the found papers into a string
        papers_str = self._format_papers(papers)

        try:
            # Second LLM request to determine which papers to cite
            text, msg_history = self._get_second_llm_response(context, papers_str, msg_history)

            if "Do not add any" in text:
                print("Do not add any.")
                return None, False
            
            # Parse the output and extract JSON
            json_output = self._extract_json_output(text)
            desc = json_output["Description"]
            selected_papers = json_output["Selected"]
            selected_papers = str(selected_papers)

            # Convert selected papers to a list of indices
            if selected_papers:
                selected_papers = self._convert_selected_papers(selected_papers, papers)
                bibtexs = [papers[i]["citationStyles"]["bibtex"] for i in selected_papers]
                bibtex_string = "\n".join(bibtexs)
            else:
                return None, False

        except Exception as e:
            print(f"Error during paper selection: {e}")
            return None, False

        # Add citation to draft
        aider_prompt = self._construct_aider_prompt(bibtex_string, desc)

        return aider_prompt, False

    def _get_initial_llm_response(self, context: CitationContext, msg_history: list) -> tuple[str, list]:
        return get_response_from_llm(
            citation_first_prompt.format(
                draft=context.draft, current_round=context.current_round, total_rounds=context.total_rounds
            ),
            client=context.client,
            model=context.model,
            system_message=citation_system_msg.format(total_rounds=context.total_rounds),
            msg_history=msg_history,
        )

    def _extract_json_output(self, text: str) -> dict:
        json_output = extract_json_between_markers(text)
        if json_output is None:
            raise ValueError("Failed to extract JSON from LLM output")
        return json_output

    def _format_papers(self, papers: list) -> str:
        paper_strings = [
            f"{i}: {paper['title']}. {paper['authors']}. {paper['venue']}, {paper['year']}. Abstract: {paper['abstract']}"
            for i, paper in enumerate(papers)
        ]
        return "\n\n".join(paper_strings)
    
    def _get_second_llm_response(self, context: CitationContext, papers_str: str, msg_history: list) -> tuple[str, list]:
        return get_response_from_llm(
            citation_second_prompt.format(
                papers=papers_str,
                current_round=context.current_round,
                total_rounds=context.total_rounds,
            ),
            client=context.client,
            model=context.model,
            system_message=citation_system_msg.format(total_rounds=context.total_rounds),
            msg_history=msg_history,
        )

    def _convert_selected_papers(self, selected_papers: str, papers: list) -> list:
        selected_papers = list(map(int, selected_papers.strip("[]").split(",")))
        if not all(0 <= i < len(papers) for i in selected_papers):
            raise IndexError("Invalid paper index")
        return selected_papers

    def _construct_aider_prompt(self, bibtex_string: str, description: str) -> str:
        aider_format = '''The following citations have just been added to the end of the `references.bib` file definition at the top of the file:
        """
        {bibtex}
        """
        You do not need to add them yourself.
        ABSOLUTELY DO NOT ADD IT AGAIN!!!

        Make the proposed change to the draft incorporating these new cites:
        {description}

        Use your judgment for whether these should be cited anywhere else.
        Make sure that any citation precisely matches the name in `references.bib`. Change its name to the correct name in the bibtex if needed.
        Ensure the citation is well-integrated into the text.'''

        return (
            aider_format.format(bibtex=bibtex_string, description=description)
            + "\n You must use \\cite or \\citet to reference papers, do not manually type out author names."
        )


class DraftImprovementComponent(ComponentBase):
    def __init__(
        self, 
        writeup_file: str, 
        exp_file: str, 
        notes: str, 
        model: str | Model, 
        io: InputOutput
    ):
        super().__init__(writeup_file, exp_file, notes, model, io)    

    def perform_improvement(self, review: str, memory_: dict[str, Any]) -> dict[str, Any]:
        improvement_prompt = '''The following review has been created for your research paper:
        """
        {review}
        """

        Improve the text using the review.'''.format(review=json.dumps(review))
        coder_out = self.coder.run(improvement_prompt)
        memory_["improved_writeup_content"] = coder_out
        memory_["is_improvement_successful"] = True
        return memory_

    def __call__(
        self,
        review: str,
        memory_: dict[str, Any],
    ) -> dict[str, Any]:
    
        # Generate PDF after the improvement process
        tex_node = LatexNode(memory_["improved_writeup_content"])
        tex_node.setup_latex_utils(self.coder)  # Use the shared Coder instance
        # // TODO: the path must be changed correctly
        template_folder = "/workspaces/researchgraph/researchgraph/graph/ai_scientist/templates/2d_diffusion"
        figures_folder = "/workspaces/researchgraph/researchgraph/graph/ai_scientist/"
        pdf_file = "/workspaces/researchgraph/data/improved_test.pdf"  

        tex_node.generate_latex(template_folder, figures_folder, pdf_file)

        return memory_
    
if __name__ == "__main__":

    from unittest.mock import MagicMock    

    # 簡単なテストのための準備
    writeup_file = "/workspaces/researchgraph/researchgraph/writingnode/sample_writeup.txt"  # テスト用のファイルパス
    exp_file = "/workspaces/researchgraph/researchgraph/writingnode/sample_exp.txt"  # 実験結果のファイルパス
    notes = "/workspaces/researchgraph/researchgraph/writingnode/sample_notes.txt"  # ノートのファイルパス
    model = Model("gpt-4-turbo")
    io = InputOutput()

    # モック化: Coder の run メソッド
    Coder.run = MagicMock(return_value="Mocked output for section generation.")

    # メモリとして利用する辞書を初期化
    memory_ = {
        "writeup_content": {},
        "is_writeup_successful": False,
        "improved_writeup_content": "",
        "is_improvement_successful": False
    }

    # アイデアとして使用するダミーデータ
    idea = {
        "title": "A Study on 2D Diffusion",
        "abstract": "This paper explores the methodology of 2D diffusion...",
        "keywords": ["diffusion", "machine learning", "simulation"]
    }

    # テスト用フォルダとクライアント
    folder_name = "/workspaces/researchgraph/researchgraph/graph/ai_scientist/templates/2d_diffusion"
    cite_client = None  # ここでは簡略化のためにNoneを使用します
    cite_model = "gpt-4-turbo"

    # WriteupComponent のインスタンス化と呼び出し
    writeup_component = WriteupComponent(writeup_file, exp_file, notes, model, io)
    print("Running WriteupComponent to generate PDF...")

    # 引用ラウンド数を1に減らして実行時間を短縮
    memory_ = writeup_component(idea, folder_name, memory_, cite_client, cite_model, num_cite_rounds=1)

    if memory_["is_writeup_successful"]:
        print("Writeup and PDF generation successful.")
    else:
        print("Writeup or PDF generation failed.")

    # DraftImprovementComponent のインスタンス化と呼び出し
    review = "The experimental setup needs more detail. Add more discussion on parameter tuning."
    improvement_component = DraftImprovementComponent(writeup_file, exp_file, notes, model, io)
    print("Running DraftImprovementComponent to improve writeup and generate PDF...")
    
    memory_ = improvement_component(review, memory_)

    # モック解除したので、LaTeXコンパイルを行う
    try:
        memory_ = improvement_component(review, memory_)
        if memory_["is_improvement_successful"]:
            print("DraftImprovementComponent test passed. Writeup improvement and PDF generation successful.")
        else:
            print("DraftImprovementComponent test failed. Writeup improvement or PDF generation failed.")
    except Exception as e:
        print(f"DraftImprovementComponent test raised an error: {e}")




# if __name__ == "__main__":
#     from aider.coders import Coder
#     from aider.models import Model
#     from aider.io import InputOutput
#     import json

#     parser = argparse.ArgumentParser(description="Perform writeup for a project")
#     parser.add_argument("--folder", type=str)
#     parser.add_argument("--no-writing", action="store_true", help="Only generate")
#     parser.add_argument(
#         "--model",
#         type=str,
#         default="gpt-4o-2024-05-13",
#         choices=[
#             "claude-3-5-sonnet-20240620",
#             "gpt-4o-2024-05-13",
#             "deepseek-coder-v2-0724",
#             "llama3.1-405b",
#             # Anthropic Claude models via Amazon Bedrock
#             "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
#             "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
#             "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
#             "bedrock/anthropic.claude-3-opus-20240229-v1:0"
#             # Anthropic Claude models Vertex AI
#             "vertex_ai/claude-3-opus@20240229",
#             "vertex_ai/claude-3-5-sonnet@20240620",
#             "vertex_ai/claude-3-sonnet@20240229",
#             "vertex_ai/claude-3-haiku@20240307"
#         ],
#         help="Model to use for AI Scientist.",
#     )
#     args = parser.parse_args()
#     if args.model == "claude-3-5-sonnet-20240620":
#         import anthropic

#         print(f"Using Anthropic API with model {args.model}.")
#         client_model = "claude-3-5-sonnet-20240620"
#         client = anthropic.Anthropic()
#     elif args.model.startswith("bedrock") and "claude" in args.model:
#         import anthropic

#         # Expects: bedrock/<MODEL_ID>
#         client_model = args.model.split("/")[-1]

#         print(f"Using Amazon Bedrock with model {client_model}.")
#         client = anthropic.AnthropicBedrock()
#     elif args.model.startswith("vertex_ai") and "claude" in args.model:
#         import anthropic

#         # Expects: vertex_ai/<MODEL_ID>
#         client_model = args.model.split("/")[-1]

#         print(f"Using Vertex AI with model {client_model}.")
#         client = anthropic.AnthropicVertex()
#     elif args.model == "gpt-4o-2024-05-13":
#         import openai

#         print(f"Using OpenAI API with model {args.model}.")
#         client_model = "gpt-4o-2024-05-13"
#         client = openai.OpenAI()
#     elif args.model == "deepseek-coder-v2-0724":
#         import openai

#         print(f"Using OpenAI API with {args.model}.")
#         client_model = "deepseek-coder-v2-0724"
#         client = openai.OpenAI(
#             api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"
#         )
#     elif args.model == "llama3.1-405b":
#         import openai

#         print(f"Using OpenAI API with {args.model}.")
#         client_model = "meta-llama/llama-3.1-405b-instruct"
#         client = openai.OpenAI(
#             api_key=os.environ["OPENROUTER_API_KEY"],
#             base_url="https://openrouter.ai/api/v1",
#         )
#     else:
#         raise ValueError(f"Model {args.model} not recognized.")
#     print("Make sure you cleaned the Aider logs if re-generating the writeup!")
#     folder_name = args.folder
#     idea_name = osp.basename(folder_name)
#     exp_file = osp.join(folder_name, "experiment.py")
#     vis_file = osp.join(folder_name, "plot.py")
#     notes = osp.join(folder_name, "notes.txt")
#     model = args.model
#     writeup_file = osp.join(folder_name, "latex", "template.tex")
#     ideas_file = osp.join(folder_name, "ideas.json")
#     with open(ideas_file, "r") as f:
#         ideas = json.load(f)
#     for idea in ideas:
#         if idea["Name"] in idea_name:
#             print(f"Found idea: {idea['Name']}")
#             break
#     if idea["Name"] not in idea_name:
#         raise ValueError(f"Idea {idea_name} not found")
#     fnames = [exp_file, writeup_file, notes]
#     io = InputOutput(yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt")
#     if args.model == "deepseek-coder-v2-0724":
#         main_model = Model("deepseek/deepseek-coder")
#     elif args.model == "llama3.1-405b":
#         main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
#     else:
#         main_model = Model(model)
#     coder = Coder.create(
#         main_model=main_model,
#         fnames=fnames,
#         io=io,
#         stream=False,
#         use_git=False,
#         edit_format="diff",
#     )
#     if args.no_writing:
#         generate_latex(coder, args.folder, f"{args.folder}/test.pdf")
#     else:
#         try:
#             perform_writeup(idea, folder_name, coder, client, client_model)
#         except Exception as e:
#             print(f"Failed to perform writeup: {e}")
