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
    def __init__(self, model: str, template_dir: str, figures_dir: str):
        # Define default files
        self.template_dir = osp.abspath(template_dir)
        self.figures_dir = figures_dir
        self.cwd = osp.join(self.template_dir, "latex")
        self.template_file = osp.join(self.cwd, "template.tex")
        self.template_copy_file = osp.join(self.cwd, "template_copy.tex")

        # Initialize paths for writeup and output PDF
        self.writeup_file_path = None
        self.pdf_file_path = None

        # Add the LaTeX template file to the list of filenames
        fnames = [self.template_copy_file]

        # Create the Coder instance
        self.coder = Coder.create(
            main_model=Model(model),
            fnames=fnames,
            io=InputOutput(),
            stream=False,
            use_git=False,
            edit_format="diff",
        )

    def set_paths(self, writeup_file_path: str, pdf_file_path: str):
        """Set the paths for the writeup file and PDF output."""
        self.writeup_file_path = writeup_file_path
        self.pdf_file_path = pdf_file_path

    def prepare_template_copy(self):
        # Copy template.tex
        shutil.copyfile(self.template_file, self.template_copy_file)

    # Check all references are valid and in the references.bib file
    def check_references(self, tex_text: str) -> bool:
        cites = re.findall(r"\\cite[a-z]*{([^}]*)}", tex_text)
        references_bib = re.search(
            r"\\begin{filecontents}{references.bib}(.*?)\\end{filecontents}",
            tex_text,
            re.DOTALL,
        )

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
    def check_figures(
        self, tex_text: str, pattern: str = r"\\includegraphics.*?{(.*?)}"
    ):
        referenced_figs = re.findall(pattern, tex_text)
        all_figs = [f for f in os.listdir(self.figures_dir) if f.endswith(".png")]

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
            check_output = os.popen(
                f"chktex {writeup_file} -q -n2 -n24 -n13 -n1"
            ).read()
            if check_output:
                prompt = f"""Please fix the following LaTeX errors in `template_copy.tex` guided by the output of `chktex`:
                {check_output}.

                Make the minimal fix required and do not remove or change any packages.
                Pay attention to any accidental uses of HTML syntax, e.g. </end instead of \\end.
                """
                self.coder.run(prompt)
            else:
                break

    def compile_latex(self, pdf_file_path: str, timeout: int = 30):
        print("GENERATING LATEX")

        commands = [
            ["pdflatex", "-interaction=nonstopmode", self.template_copy_file],
            ["bibtex", osp.splitext(self.template_copy_file)[0]],
            ["pdflatex", "-interaction=nonstopmode", self.template_copy_file],
            ["pdflatex", "-interaction=nonstopmode", self.template_copy_file],
        ]

        for command in commands:
            try:
                result = subprocess.run(
                    command,
                    cwd=self.cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                )
                print("Standard Output:\n", result.stdout)
                print("Standard Error:\n", result.stderr)
            except subprocess.TimeoutExpired as e:
                print(f"Latex command timed out: {e}")
            except subprocess.CalledProcessError as e:
                print(f"Error running command {' '.join(command)}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

        print("FINISHED GENERATING LATEX")

        pdf_filename = f"{osp.splitext(osp.basename(self.template_copy_file))[0]}.pdf"
        try:
            shutil.move(osp.join(self.cwd, pdf_filename), self.pdf_file_path)
        except FileNotFoundError:
            print("Failed to rename PDF.")


class LatexNode:
    def __init__(
        self,
        input_variable,
        output_variable,
        model: str,
        template_dir: str,
        figures_dir: str,
        timeout: int = 30,
        num_error_corrections: int = 5,
    ):
        self.input_variable = input_variable
        self.output_variable = output_variable
        self.latex_utils = LatexUtils(model, template_dir, figures_dir)
        self.timeout = timeout
        self.num_error_corrections = num_error_corrections

    def __call__(self, state: State) -> dict:
        try:
            # Get paths from state
            writeup_file_path = osp.expanduser(state.get(self.input_variable))
            pdf_file_path = osp.expanduser(state.get(self.output_variable))

            if not writeup_file_path or not pdf_file_path:
                raise ValueError("Input or output file path not found in state.")

            # Set paths in LatexUtils instance
            self.latex_utils.set_paths(writeup_file_path, pdf_file_path)

            # Prepare LaTeX template copy
            self.latex_utils.prepare_template_copy()

            tex_text = ""

            try:
                with open(writeup_file_path, "r") as f:
                    file_content = f.read()
            except FileNotFoundError:
                print(f"Writeup file '{writeup_file_path}' not found.")
                return None
            except PermissionError:
                print(f"Permission denied to read '{writeup_file_path}'.")
                return None

            # Split file content into sections based on headings (e.g., "# abstract", "# introduction")
            sections = self._split_into_sections(file_content)

            # Read the LaTeX template content
            with open(self.latex_utils.template_copy_file, "r") as f:
                tex_text = f.read()

            # Replace placeholders in the LaTeX template with corresponding section content
            for section, content in sections.items():
                placeholder = f"{section.upper()} HERE"
                tex_text = tex_text.replace(placeholder, content)

            with open(self.latex_utils.template_copy_file, "w") as f:
                f.write(tex_text)

            # Run LaTeX utilities
            self.latex_utils.check_references(tex_text)
            self.latex_utils.check_figures(tex_text)
            self.latex_utils.check_duplicates(
                tex_text, r"\\includegraphics.*?{(.*?)}", "figure"
            )
            self.latex_utils.check_duplicates(
                tex_text, r"\\section{([^}]*)}", "section header"
            )
            self.latex_utils.fix_latex_errors(
                self.latex_utils.template_copy_file, self.num_error_corrections
            )

            self.latex_utils.compile_latex(pdf_file_path, timeout=self.timeout)

            # Update state with output PDF path
            return {self.output_variable: pdf_file_path}

        except Exception as e:
            print(f"Error occurred: {e}")
            return None

    def _split_into_sections(self, text: str) -> dict:
        # Split the text into sections based on headings (e.g., "# abstract", "# introduction")
        sections = {}
        current_section = None
        buffer = []

        for line in text.splitlines():
            match = re.match(
                r"#\s*([\w\s]+)", line
            )  # TODO: プレーンテキストに対するセクションの判定条件
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
        model=model,
        template_dir=template_dir,
        figures_dir=figures_dir,
        timeout=30,
        num_error_corrections=5,
    )

    # Create the StateGraph and add node
    graph_builder = StateGraph(State)
    graph_builder.add_node("latexnode", latex_node)
    graph_builder.set_entry_point("latexnode")
    graph_builder.set_finish_point("latexnode")
    graph = graph_builder.compile()

    # Define initial state
    memory = {
        "writeup_file_path": "/workspaces/researchgraph/data/writeup_file.txt",
        "pdf_file_path": "/workspaces/researchgraph/data/sample.pdf",
    }

    # Execute the graph
    graph.invoke(memory)
