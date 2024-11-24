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
from langgraph.graph import StateGraph
from typing import TypedDict

from researchgraph.graph.ai_scientist.ai_scientist_node.prompt.writeup_prompt import (
    per_section_tips,
    error_list,
    citation_system_msg,
    citation_first_prompt,
    citation_second_prompt, 
    prompt_templates, 
    related_work_prompt, 
)


class State(TypedDict):
    notes_path: str
    writeup_file_path: str
    review_path: str | None     # DraftImprovementComponent

@dataclass
class CitationContext:
    client: Any
    model: str
    draft: str
    current_round: int
    total_rounds: int


class CitationManager:
    def __init__(self, coder: Coder):
        self.coder = coder

    def add_citations(self, sections: dict, template_dir: str, cite_client: Any, cite_model: Model, num_cite_rounds: int):
            for round_num in range(num_cite_rounds):
                template_file = osp.join(template_dir, "latex", "template.tex")
                with open(template_file, "r") as f:
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

                    with open(template_file, "w") as f:
                        f.write(draft)

                self.coder.run(prompt)      # TODO: 執筆内容に引用情報を付与する

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


class BaseSection:
    def __init__(self, coder: Coder, section_name: str):
        self.coder = coder
        self.section_name = section_name
        self.content = ""

    def generate_prompt(self, prompt_type: str) -> str:
        if prompt_type not in prompt_templates:
            raise ValueError("Invalid prompt type specified.")
        
        prompt = prompt_templates[prompt_type]
        
        prompt = prompt.format(
            section_name=self.section_name,
            tips=per_section_tips.get(self.section_name, ""),
            error_list=error_list
        )
        
        return prompt.replace(r"{{", "{").replace(r"}}", "}")

    def write(self):
        prompt = self.generate_prompt("write")
        self.content = self.coder.run(prompt)

    def refine(self):
        prompt = self.generate_prompt("refine")
        self.content = self.coder.run(prompt)

    def second_refine(self) -> Any:
        prompt = self.generate_prompt("second_refine")
        self.content = self.coder.run(prompt)
        

class RelatedWorkSection(BaseSection):
    def __init__(self, coder: Coder):
        super().__init__(coder, "Related Work")

    def generate_prompt(self, prompt_type) -> str:
        if prompt_type == "write":
            prompt = related_work_prompt["write"].format(
                tips=per_section_tips["Related Work"]
            )
            return prompt

        return super().generate_prompt(prompt_type)


class WriteupComponent:
    def __init__(
        self, 
        input_variable: str,   # notes_path
        output_variable: str,   # writeup_file_path
        model: str,
        template_dir: str, 
        cite_client: Any, 
        num_cite_rounds: int,         
    ):
        self.input_variable = input_variable
        self.output_variable = output_variable
        self.model = model
        self.template_dir = template_dir
        self.cite_client = cite_client 
        self.num_cite_rounds = num_cite_rounds
        self.sections = None
        self.citation_manager = None

    # PERFORM WRITEUP
    def __call__(self, state: State) -> dict:
        notes_path = state[self.input_variable]
        writeup_file_path = state[self.output_variable]

        # Check if the notes file exists, raise an error if it doesn't
        if not os.path.exists(notes_path):
            raise FileNotFoundError(f"The specified notes file does not exist: {notes_path}")
        
        main_model = self._select_model(self.model)

        self.coder = Coder.create(
            main_model=main_model,
            fnames=[notes_path],     
            io=InputOutput(),
            stream=False,
            use_git=False,
            edit_format="diff",
        )

        if self.sections is None:
            self.sections = {
                "Title": BaseSection(self.coder, "Title"),
                "Abstract": BaseSection(self.coder, "Abstract"),
                "Introduction": BaseSection(self.coder, "Introduction"),
                "Background": BaseSection(self.coder, "Background"),
                "Method": BaseSection(self.coder, "Method"),
                "Experimental Setup": BaseSection(self.coder, "Experimental Setup"),
                "Results": BaseSection(self.coder, "Results"),
                "Conclusion": BaseSection(self.coder, "Conclusion"),
                "Related Work": RelatedWorkSection(self.coder),
            }

        # Write each section directly to the output file
        with open(writeup_file_path, "w") as f:
            for section_name, section in self.sections.items():
                # Step 1: Initial Writing
                section.write()
                # Step 2: First Refinement (remove placeholders and improve quality)
                section.refine()
                # Step 3: Second Refinement (final touches, clean up any remaining meta information)
                section.second_refine()
                
                # Write the final refined content to the file without any metadata or placeholders
                f.write(f"# {section_name}\n")
                f.write(section.content.strip() + "\n")

        # CitationManager インスタンスの初期化
        if self.citation_manager is None:
            self.citation_manager = CitationManager(self.coder)       

        # Add citations to each section after refinement
        cite_model = self.model
        self.citation_manager.add_citations(self.sections, self.template_dir, self.cite_client, cite_model, self.num_cite_rounds)

        # Save the final write-up content after adding citations
        with open(writeup_file_path, "a") as f:
            for section in self.sections.values():
                f.write(section.content + "\n")

        return {
            self.output_variable: writeup_file_path
        }

    def _select_model(self, model: str) -> Model:
        model_mapping = {
            "deepseek-coder-v2-0724": "deepseek/deepseek-coder",
            "llama3.1-405b": "openrouter/meta-llama/llama-3.1-405b-instruct"
        }
        return Model(model_mapping.get(model, model))


class DraftImprovementComponent:
    def __init__(
        self, 
        input_variable: list,   # writeup_file, notes, review_path
        output_variable: str,   # 
        model: str, 
        io: InputOutput
    ):
        self.input_variable = input_variable
        self.output_variable = output_variable
        self.coder = Coder.create(
            main_model=model,
            fnames=input_variable,
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )

    def __call__(self, state: State) -> dict:
        # Extract review content from state
        review_path = state.get("review_path")
        if not review_path:
            raise ValueError("Review content is required in the state.")
        
        with open(review_path, "r") as f:
                review_content = f.read()
    
        # Perform improvement using the review
        improved_content = self.perform_improvement(review_content)

        # Save the perform_improvement content to the output file
        output_file = self.output_variable
        with open(output_file, "w") as f:
            f.write(improved_content + "\n")

        return {
            self.output_variable: output_file
        }

    def perform_improvement(self, review: str) -> str:
        improvement_prompt = '''The following review has been created for your research paper:
        """
        {review}
        """

        Improve the text using the review.'''.format(review=json.dumps(review))
        coder_out = self.coder.run(improvement_prompt)
        return coder_out
    

if __name__ == "__main__":

    import openai

    # Define input and output variables
    input_variable = "notes_path"
    output_variable = "writeup_file_path"
    model = "gpt-3.5-turbo"
    io = InputOutput()
    template_dir = "/workspaces/researchgraph/researchgraph/graph/ai_scientist/templates/2d_diffusion"
    cite_client = openai

    # Initialize WriteupComponent as a LangGraph node
    writeup_component = WriteupComponent(
        input_variable=input_variable,
        output_variable=output_variable,
        model=model,
        template_dir=template_dir, 
        cite_client=cite_client, 
        num_cite_rounds=2, 
    )

    # Create the StateGraph and add node
    graph_builder = StateGraph(State)
    graph_builder.add_node("writeup_component", writeup_component)
    graph_builder.set_entry_point("writeup_component")
    graph_builder.set_finish_point("writeup_component")
    graph = graph_builder.compile()

    # Define initial state
    memory = {
        "notes_path": "/workspaces/researchgraph/data/notes.txt", 
        "writeup_file_path": "/workspaces/researchgraph/data/writeup_file.txt"
    }

    # Execute the graph
    graph.invoke(memory)





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
