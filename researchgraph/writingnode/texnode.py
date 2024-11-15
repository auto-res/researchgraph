import os
import os.path as osp
import re
import subprocess
import shutil
from typing import TypedDict
from langgraph.graph import StateGraph
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput


class State(TypedDict):
    writeup_file_path: str
    pdf_file_path: str


class LatexUtils:
    def __init__(self, model: str):
        # Define default files
        fnames = []
        # Create the Coder instance
        self.coder = Coder.create(      # TODO: インスタンス引数
            main_model=Model(model),
            fnames=fnames,
            io=InputOutput(),
            stream=False,
            use_git=False,
            edit_format="diff",
        )

    # Check all references are valid and in the references.bib file
    def check_references(self, tex_text: str) -> bool:
        cites = re.findall(r"\\cite[a-z]*{([^}]*)}", tex_text)
        references_bib = re.search(r"\\begin{filecontents}{references.bib}(.*?)\\end{filecontents}", tex_text, re.DOTALL)

        if references_bib is None:
            print("No references.bib found in template_copy.tex")
            return False
        
        bib_text = references_bib.group(1)
        missing_cites = [cite for cite in cites if cite.strip() not in bib_text]

        for cite in missing_cites:
            print(f"Reference {cite} not found in references.")
            self._prompt_fix_reference(cite)

        return True

    def _prompt_fix_reference(self, cite: str):
        prompt = f"""Reference {cite} not found in references.bib. Is this included under a different name?
        If so, please modify the citation in template_copy.tex to match the name in references.bib at the top. Otherwise, remove the cite."""
        self.coder.run(prompt)

    # Check all included figures are actually in the directory.
    def check_figures(self, figure_dir: str, tex_text: str, pattern: str = r"\\includegraphics.*?{(.*?)}"):
        referenced_figs = re.findall(pattern, tex_text)
        all_figs = [f for f in os.listdir(figure_dir) if f.endswith(".png")]

        for fig in referenced_figs:
            if fig not in all_figs:
                print(f"Figure {fig} not found in directory.")
                self._prompt_fix_figure(fig, all_figs)

    def _prompt_fix_figure(self, fig: str, all_figs: list):
        if not all_figs:
            prompt = f"""The image {fig} not found in the directory and there are no images present. Please add the required image to the directory, ensuring the filename matches {fig}. Refer to the project documentation or notes for details on what the figure should contain."""
        else:
            prompt = f"""The image {fig} not found in the directory. The images in the directory are: {all_figs}.
            Please ensure that the figure is in the directory and that the filename is correct. Check the notes to see what each figure contains."""
        self.coder.run(prompt)

    # Remove duplicate items.
    def check_duplicates(self, tex_text: str, pattern: str, element_type: str):
        items = re.findall(pattern, tex_text)
        duplicates = {x for x in items if items.count(x) > 1}

        for dup in duplicates:
            print(f"Duplicate {element_type} found: {dup}.")
            self._prompt_fix_duplicates(dup, element_type)

    def _prompt_fix_duplicates(self, dup: str, element_type: str):
        prompt = f"""Duplicate {element_type} found: {dup}. Ensure any {element_type} is only included once.
        If duplicated, identify the best location for the {element_type} and remove any other."""
        self.coder.run(prompt)

    # Iteratively fix any LaTeX bugs
    def fix_latex_errors(self, writeup_file: str, num_error_corrections: int = 5):
        for _ in range(num_error_corrections):
            check_output = os.popen(f"chktex {writeup_file} -q -n2 -n24 -n13 -n1").read()
            if check_output:
                prompt = f"""Please fix the following LaTeX errors in `template_copy.tex` guided by the output of `chktek`:
                {check_output}.

                Make the minimal fix required and do not remove or change any packages.
                Pay attention to any accidental uses of HTML syntax, e.g. </end instead of \\end.
                """
                self.coder.run(prompt)
            else:
                break

    def compile_latex(self, cwd: str, pdf_file_path: str, template_file: str, timeout: int = 30):
        print("GENERATING LATEX")

        commands = [
            ["pdflatex", "-interaction=nonstopmode", template_file],
            ["bibtex", osp.splitext(template_file)[0]], 
            ["pdflatex", "-interaction=nonstopmode", template_file],
            ["pdflatex", "-interaction=nonstopmode", template_file],
        ]

        for command in commands:
            try:
                result = subprocess.run(
                    command,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                )
                print("Standard Output:\n", result.stdout)
                print("Standard Error:\n", result.stderr)
            except subprocess.TimeoutExpired:
                print(f"Latex timed out after {timeout} seconds")
            except subprocess.CalledProcessError as e:
                print(f"Error running command {' '.join(command)}: {e}")

        print("FINISHED GENERATING LATEX")

        pdf_filename = f"{osp.splitext(osp.basename(template_file))[0]}.pdf"
        try:
            shutil.move(osp.join(cwd, pdf_filename), pdf_file_path)
        except FileNotFoundError:
            print("Failed to rename PDF.")


class LatexNode:
    def __init__(self, input_variable, output_variable, model: str, template_dir: str, figures_dir: str, timeout: int = 30, num_error_corrections: int = 5):
        self.input_variable = input_variable
        self.output_variable = output_variable
        self.latex_utils = LatexUtils(model)
        self.template_dir = template_dir
        self.figures_dir = figures_dir
        self.timeout = timeout
        self.num_error_corrections = num_error_corrections

    def __call__(self, state: State) -> dict:
        try:
            # Generate LaTeX content
            template_dir = osp.abspath(self.template_dir)
            cwd = osp.join(template_dir, "latex")
            template_file = osp.join(cwd, "template.tex")

            # Copy template.tex
            template_copy_file = osp.join(cwd, "template_copy.tex")
            shutil.copyfile(template_file, template_copy_file)

            tex_text = ''

            # Read content from input_variable path
            writeup_file_path = state.get(self.input_variable)
            if not writeup_file_path:
                raise ValueError(f"Input file path for variable '{self.input_variable}' not found in state.")
            
            with open(writeup_file_path, "r") as f:
                file_content = f.read()

            # Split file content into sections based on headings (e.g., "# abstract", "# introduction")
            sections = self._split_into_sections(file_content)

            # Read the LaTeX template content
            with open(template_copy_file, "r") as f:
                tex_text = f.read()

            # Replace placeholders in the LaTeX template with corresponding section content
            for section, content in sections.items():
                placeholder = f"{section.upper()} HERE"
                tex_text = tex_text.replace(placeholder, content)

            with open(template_copy_file, "w") as f:
                f.write(tex_text)

            # Run LaTeX utilities
            self.latex_utils.check_references(tex_text)
            self.latex_utils.check_figures(self.figures_dir, tex_text)
            self.latex_utils.check_duplicates(tex_text, r"\\includegraphics.*?{(.*?)}", "figure")
            self.latex_utils.check_duplicates(tex_text, r"\\section{([^}]*)}", "section header")
            self.latex_utils.fix_latex_errors(template_copy_file, self.num_error_corrections)

            # Compile LaTeX to PDF
            pdf_file_path = state.get(self.output_variable)
            if not pdf_file_path:
                raise ValueError(f"Output file path for variable '{self.output_variable}' not found in state.")
            
            self.latex_utils.compile_latex(cwd, pdf_file_path, template_copy_file, timeout=self.timeout)

            # Update state with output PDF path
            return {
                self.output_variable: pdf_file_path
            }

        except Exception as e:
            print(f"Error occurred: {e}")
            return None
        
    def _split_into_sections(self, text: str) -> dict:
        # Split the text into sections based on headings (e.g., "# abstract", "# introduction")
        sections = {}
        current_section = None
        buffer = []

        for line in text.splitlines():
            match = re.match(r"#\s*(\w+)", line)    # TODO: プレーンテキストに対するセクションの判定条件
            if match:
                if current_section and buffer:
                    sections[current_section] = "\n".join(buffer).strip()
                    buffer = []
                current_section = match.group(1).lower()
            elif current_section:
                buffer.append(line)

        if current_section and buffer:
            sections[current_section] = "\n".join(buffer).strip()

        return sections


if __name__ == "__main__":

    # Define input and output variables
    input_variable = "writeup_file_path" 
    output_variable = "pdf_file_path"
    model = "gpt-4o"
    template_dir = "/workspaces/researchgraph/researchgraph/graph/ai_scientist/templates/2d_diffusion"
    figures_dir = "/workspaces/researchgraph/images"

    # Initialize LatexNode
    latex_node = LatexNode(
        input_variable=input_variable,
        output_variable=output_variable,
        model = model, 
        template_dir=template_dir,
        figures_dir=figures_dir,
        timeout=30,
        num_error_corrections=5
    )

    # Create the StateGraph and add node
    graph_builder = StateGraph(State)
    graph_builder.add_node("latexnode", latex_node)
    graph_builder.set_entry_point("latexnode")
    graph_builder.set_finish_point("latexnode")
    graph = graph_builder.compile()

    # Define initial state
    memory = {
        "writeup_file_path" : "~/write_file.txt",
        "pdf_file_path": "~/data/sample.pdf"
    }

    # Execute the graph
    graph.invoke(memory)
