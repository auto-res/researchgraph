import os
import os.path as osp
import re
import subprocess
import shutil
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
from researchgraph.core.node import Node


class LatexNode(Node):
    def __init__(
        self,
        input_key: list[str],
        output_key: list[str],
        model: str,
        template_dir: str,
        figures_dir: str,
        timeout: int = 30,
        num_error_corrections: int = 5,
    ):
        super().__init__(input_key, output_key)
        self.timeout = timeout
        self.num_error_corrections = num_error_corrections
        self.figures_dir = figures_dir
        self.template_file = osp.join(osp.abspath(template_dir), "latex", "template.tex")
        self.template_copy_file = osp.join(osp.abspath(template_dir), "latex", "template_copy.tex")

        self.coder = Coder.create(
            main_model=Model(model),
            fnames=[],
            io=InputOutput(),
            stream=False,
            use_git=False,
            edit_format="diff",
        )

    def _copy_template(self):
        # Copy the LaTeX template to a working copy for modifications
        if not osp.exists(self.template_file):
            raise FileNotFoundError(f"Template file not found: {self.template_file}")
        shutil.copyfile(self.template_file, self.template_copy_file)

    def _fill_template(self, content: dict) -> str:
        # Read the copied template, replace placeholders with content, and save the updated file
        tex_text = ""
        with open(self.template_copy_file, "r") as f:
            tex_text = f.read()

        for section, value in content.items():
            placeholder = f"{section.upper()} HERE"
            tex_text = tex_text.replace(placeholder, value)

        with open(self.template_copy_file, "w") as f:
            f.write(tex_text)
        return tex_text

    def _check_references(self, tex_text: str) -> bool:
        # Check for missing references in the LaTeX content against the references.bib section
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
            prompt = f"""Reference {cite} not found in references.bib. Is this included under a different name?
            If so, please modify the citation in template_copy.tex to match the name in references.bib at the top. Otherwise, remove the cite."""
            self.coder.run(prompt)

    def _check_figures(
        self,
        tex_text: str,
        figures_dir: str,
        pattern: str = r"\\includegraphics.*?{(.*?)}",
    ):
        # Verify all referenced figures in the LaTeX content exist in the figures directory
        referenced_figs = re.findall(pattern, tex_text)
        all_figs = [f for f in os.listdir(figures_dir) if f.endswith(".png")]

        for fig in referenced_figs:
            if fig not in all_figs:
                print(f"Figure {fig} not found in directory.")
                if not all_figs:
                    prompt = f"""The image {fig} not found in the directory and there are no images present. Please add the required image to the directory, ensuring the filename matches {fig}. Refer to the project documentation or notes for details on what the figure should contain."""
                else:
                    prompt = f"""The image {fig} not found in the directory. The images in the directory are: {all_figs}.
                    Please ensure that the figure is in the directory and that the filename is correct. Check the notes to see what each figure contains."""
                self.coder.run(prompt)

    def _check_duplicates(self, tex_text: str, patterns: dict):
        # Detect and prompt for duplicate elements in the LaTeX content
        for element_type, pattern in patterns.items():
            items = re.findall(pattern, tex_text)
            duplicates = {x for x in items if items.count(x) > 1}
            for dup in duplicates:
                print(f"Duplicate {element_type} found: {dup}.")
                prompt = f"""Duplicate {element_type} found: {dup}. Ensure any {element_type} is only included once.
                If duplicated, identify the best location for the {element_type} and remove any other."""
                self.coder.run(prompt)

    def _fix_latex_errors(self, writeup_file: str, num_error_corrections: int = 5):
        # Fix LaTeX errors iteratively using chktex and automated suggestions
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

    def _compile_latex(
        self, cwd: str, template_copy_file: str, pdf_file_path: str, timeout: int = 30
    ):
        # Compile the LaTeX document to PDF using pdflatex and bibtex commands
        print("GENERATING LATEX")
        commands = [
            ["pdflatex", "-interaction=nonstopmode", template_copy_file],
            ["bibtex", osp.splitext(template_copy_file)[0]],
            ["pdflatex", "-interaction=nonstopmode", template_copy_file],
            ["pdflatex", "-interaction=nonstopmode", template_copy_file],
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
            except subprocess.TimeoutExpired as e:
                print(f"Latex command timed out: {e}")
            except subprocess.CalledProcessError as e:
                print(f"Error running command {' '.join(command)}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

        print("FINISHED GENERATING LATEX")
        pdf_filename = f"{osp.splitext(osp.basename(template_copy_file))[0]}.pdf"
        try:
            shutil.move(osp.join(cwd, pdf_filename), pdf_file_path)
        except FileNotFoundError:
            print("Failed to rename PDF.")

    def execute(self, state) -> dict:
        try:
            paper_content = state.get(self.input_key[0])
            pdf_file_path = osp.expanduser(state.get(self.output_key[0]))

            if not paper_content or not pdf_file_path:
                raise ValueError("Input paper content or output file path not found in state.")
            
            self._copy_template()
            tex_text = self._fill_template(paper_content)
            self._check_references(tex_text)
            self._check_figures(tex_text, self.figures_dir)
            self._check_duplicates(tex_text, {
                "figure": r"\\includegraphics.*?{(.*?)}",
                "section header": r"\\section{([^}]*)}",
            })
            self._fix_latex_errors(self.template_copy_file, self.num_error_corrections)
            self._compile_latex(
                osp.dirname(self.template_file),
                self.template_copy_file,
                pdf_file_path,
                timeout=self.timeout,
            )
            return {self.output_key[0]: pdf_file_path}

        except Exception as e:
            print(f"Error occurred: {e}")
            return None
