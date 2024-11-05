import os
import os.path as osp
import re
import subprocess
import shutil
from typing import Any
from pydantic import BaseModel
from langgraph.graph import StateGraph
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput


class State(BaseModel):
    generated_content: str = ""


class LatexUtils:
    def __init__(self, coder: Coder):
        self.coder = coder

    # Check all references are valid and in the references.bib file
    def check_references(self, tex_text: str) -> bool:
        cites = re.findall(r"\\cite[a-z]*{([^}]*)}", tex_text)
        references_bib = re.search(r"\\begin{filecontents}{references.bib}(.*?)\\end{filecontents}", tex_text, re.DOTALL)

        if references_bib is None:
            print("No references.bib found in template.tex")
            return False
        
        bib_text = references_bib.group(1)
        missing_cites = [cite for cite in cites if cite.strip() not in bib_text]

        for cite in missing_cites:
            print(f"Reference {cite} not found in references.")
            self._prompt_fix_reference(cite)

        return True

    def _prompt_fix_reference(self, cite: str):
        prompt = f"""Reference {cite} not found in references.bib. Is this included under a different name?
        If so, please modify the citation in template.tex to match the name in references.bib at the top. Otherwise, remove the cite."""
        self.coder.run(prompt)

    # Check all included figures are actually in the directory.
    def check_figures(self, folder: str, tex_text: str, pattern: str = r"\\includegraphics.*?{(.*?)}"):
        referenced_figs = re.findall(pattern, tex_text)
        all_figs = [f for f in os.listdir(folder) if f.endswith(".png")]

        for fig in referenced_figs:
            if fig not in all_figs:
                print(f"Figure {fig} not found in directory.")
                self._prompt_fix_figure(fig, all_figs)

    def _prompt_fix_figure(self, fig: str, all_figs: list):
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
                prompt = f"""Please fix the following LaTeX errors in `template.tex` guided by the output of `chktek`:
        {check_output}.

        Make the minimal fix required and do not remove or change any packages.
        Pay attention to any accidental uses of HTML syntax, e.g. </end instead of \\end.
        """
                self.coder.run(prompt)
            else:
                break

    def compile_latex(self, cwd: str, pdf_file: str, template_file: str = "template_copy.tex", timeout: int = 30):
        print("GENERATING LATEX")

        commands = [
            ["pdflatex", "-interaction=nonstopmode", template_file],
            ["bibtex", "template"],
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

        try:
            shutil.move(osp.join(cwd, "template.pdf"), pdf_file)
        except FileNotFoundError:
            print("Failed to rename PDF.")


class TextNode:
    def __init__(self, coder_out: dict[str, Any]):
        self.coder_out = coder_out

    def setup_latex_utils(self, coder: Coder):
        self.latex_utils = LatexUtils(coder)

    def generate_latex(self, folder_name: str, pdf_file: str, timeout: int = 30, num_error_corrections: int = 5):
        if not self.latex_utils:
            raise ValueError("LatexUtils not set up. Please call setup_latex_utils first.")

        folder = osp.abspath(folder_name)
        cwd = osp.join(folder, "latex")  # Fixed potential issue with path
        writeup_file = osp.join(cwd, "template.tex")

        # Copy template.tex
        writeup_copy_file = osp.join(cwd, "template_copy.tex")
        shutil.copyfile(writeup_file, writeup_copy_file)

        if self.coder_out:
            with open(writeup_copy_file, "r") as f:
                tex_text = f.read()

            for section, content in self.coder_out.items():
                placeholder = f"{section.upper()} HERE"
                tex_text = tex_text.replace(placeholder, content)

        with open(writeup_copy_file, "w") as f:
            f.write(tex_text)

        self.latex_utils.check_references(tex_text)
        self.latex_utils.check_figures(folder, tex_text)
        self.latex_utils.check_duplicates(tex_text, r"\\includegraphics.*?{(.*?)}", "figure")
        self.latex_utils.check_duplicates(tex_text, r"\\section{([^}]*)}", "section header")
        self.latex_utils.fix_latex_errors(writeup_copy_file, num_error_corrections)
        self.latex_utils.compile_latex(cwd, pdf_file, timeout=timeout)

    def __call__(self, state: State) -> State:
        try:
            generated_content = "\n".join(self.coder_out.values())
            state.generated_content = f"Generated Content:\n{generated_content}"
            return state
        
        except Exception as e:
            print(f"Error occured: {e}")
            return None    

if __name__ == "__main__":

    main_model = Model("gpt-4-turbo")
    io = InputOutput()

    coder = Coder(main_model=main_model, io=io)
    coder_out = {
        "abstract": "This is the abstract content.",
        "introduction": "This is the introduction content.",
        "method": "This is the method content."
    }

    graph_builder = StateGraph(State)
    text_node = TextNode(coder_out)
    text_node.setup_latex_utils(coder)

    graph_builder.add_node("textnode", text_node)
    graph_builder.set_entry_point("textnode")
    graph_builder.set_finish_point("textnode")
    graph = graph_builder.compile()

    memory = {
        "generated_content": ""
    }

    # graph.invoke(memory, debug=True)
    graph.invoke(memory)