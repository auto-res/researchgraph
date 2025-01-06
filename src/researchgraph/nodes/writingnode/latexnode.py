import os
import os.path as osp
import re
import subprocess
import shutil
from litellm import completion
from researchgraph.core.node import Node


class LatexNode(Node):
    def __init__(
        self,
        input_key: list[str],
        output_key: list[str],
        llm_name: str,
        template_dir: str,
        figures_dir: str,
        timeout: int = 30,
    ):
        super().__init__(input_key, output_key)
        self.llm_name = llm_name
        self.timeout = timeout
        self.figures_dir = figures_dir
        self.template_file = osp.join(osp.abspath(template_dir), "latex", "template.tex")
        self.template_copy_file = osp.join(osp.abspath(template_dir), "latex", "template_copy.tex")

    def _call_llm(self, prompt: str) -> str:
        try:
            response = completion(
                model=self.llm_name, 
                messages=[
                    {"role": "user", "content": prompt}
                ], 
                temperature=0,
            )
            output = response.choices[0].message.content
            return output
        except Exception as e:
            print("Error calling LLM: {e}")
            raise

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

    def _check_references(self, tex_text: str) -> str:
        # Check for missing references in the LaTeX content against the references.bib section
        cites = re.findall(r"\\cite[a-z]*{([^}]*)}", tex_text)
        references_bib = re.search(
            r"\\begin{filecontents}{references.bib}(.*?)\\end{filecontents}",
            tex_text,
            re.DOTALL,
        )
        if references_bib is None:
            raise FileNotFoundError("references.bib not found in template_copy.tex")
        
        bib_text = references_bib.group(1)
        missing_cites = [cite for cite in cites if cite.strip() not in bib_text]
        if not missing_cites:
            print("No missing references found.")
            return  tex_text

        print(f"Missing references found: {missing_cites}")
        prompt = f""""
        LaTeX text:
        {tex_text}

        
        References.bib content:
        {bib_text}


        The following reference is missing from references.bib: {missing_cites}.
        Please provide the complete corrected LaTeX text, ensuring the reference issue is fixed.
        
        Return the complete LaTeX document, including any bibtex changes.
        """
        llm_response = self._call_llm(prompt)
        if not llm_response:
            raise RuntimeError(f"LLM failed to respond for missing references: {missing_cites}")
        return llm_response

    def _check_figures(
        self,
        tex_text: str,
        figures_dir: str,
        pattern: str = r"\\includegraphics.*?{(.*?)}",
    ) -> str:
        # Verify all referenced figures in the LaTeX content exist in the figures directory
        referenced_figs = re.findall(pattern, tex_text)
        all_figs = [f for f in os.listdir(figures_dir) if f.endswith(".png")]

        missing_figs = [fig for fig in referenced_figs if fig not in all_figs]
        if not missing_figs:
            print("No missing figures found.")
            return tex_text

        print(f"Missin figures found: {missing_figs}")
        prompt = f"""
        LaTeX text:
        {tex_text}


        The following images were not found in the directory {figures_dir}: {', '.join(missing_figs)}.
        Available images are: {all_figs}.
        Please provide the complete corrected LaTeX text, ensuring the figure issue is fixed.

        Return the complete LaTeX document.
        """
        llm_response = self._call_llm(prompt)
        if not llm_response:
            raise RuntimeError(f"LLM failed to respond for missing figures: {missing_figs}")
        return llm_response

    def _check_duplicates(self, tex_text: str, patterns: dict) -> str:
        # Detect and prompt for duplicate elements in the LaTeX content
        for element_type, pattern in patterns.items():
            items = re.findall(pattern, tex_text)
            duplicates = {x for x in items if items.count(x) > 1}
            if duplicates:
                print(f"Duplicate {element_type} found: {duplicates}.")
                prompt = f"""
                LaTeX text:
                {tex_text}


                Duplicate {element_type} found: {', '.join(duplicates)}. Ensure any {element_type} is only included once. 
                If duplicated, identify the best location for the {element_type} and remove any other.
                
               Return the complete corrected LaTeX text with the duplicates fixed.
                """
                llm_response = self._call_llm(prompt)
                if not llm_response:
                    raise RuntimeError(f"LLM failed to respond for missing figures: {duplicates}")
                tex_text = llm_response
            return tex_text


    def _fix_latex_errors(self, writeup_file: str) -> str:
        # Fix LaTeX errors iteratively using chktex and automated suggestions
        with open(writeup_file, "r") as f:
            tex_text = f.read()
        check_output = os.popen(
            f"chktex {writeup_file} -q -n2 -n24 -n13 -n1"
        ).read()
        
        if check_output:
            error_messages = check_output.strip().split("\n")
            formatted_errors = "\n".join(
                f"- {msg}" for msg in error_messages if msg
            )
            prompt = f"""
            LaTeX text:
            {tex_text}


            Please fix the following LaTeX errors: {formatted_errors}.      
            Make the minimal fix required and do not remove or change any packages unnecessarily.
            Pay attention to any accidental uses of HTML syntax, e.g. </end instead of \\end.

            Return the complete corrected LaTeX text.
            """
            llm_response = self._call_llm(prompt)

            if not llm_response:
                raise RuntimeError(f"LLM failed to fix latex errors for {writeup_file}")
            return llm_response
        else:
            print("No LaTex errors found by chktex.")
            return tex_text

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
            paper_content = getattr(state, self.input_key[0])
            pdf_file_path = osp.expanduser(getattr(state, self.output_key[0]))

            if not paper_content or not pdf_file_path:
                raise ValueError("Input paper content or output file path not found in state.")
            
            self._copy_template()
            tex_text = self._fill_template(paper_content)

            max_iterations = 10
            iteration_count = 0

            while iteration_count < max_iterations:
                print(f"Start iteration: {iteration_count}")
                original_tex_text = tex_text
                tex_text = self._check_references(tex_text)
                if tex_text != original_tex_text:
                    iteration_count += 1
                    continue

                original_tex_text = tex_text
                tex_text = self._check_figures(tex_text, self.figures_dir)
                if tex_text != original_tex_text:
                    iteration_count += 1
                    continue

                original_tex_text = tex_text
                tex_text = self._check_duplicates(tex_text, {
                "figure": r"\\includegraphics.*?{(.*?)}",
                "section header": r"\\section{([^}]*)}",
                })
                if tex_text != original_tex_text:
                    iteration_count += 1
                    continue

                original_tex_text = tex_text
                tex_text = self._fix_latex_errors(self.template_copy_file)
                if tex_text != original_tex_text:
                    iteration_count += 1
                    continue

                print("No changes detected, exiting loop.")
                break

            if iteration_count == max_iterations:
                print(f"Maximum iterations reached ({max_iterations}), exiting loop.")

            with open(self.template_copy_file, "w") as f:
                f.write(tex_text)

            self._compile_latex(
                osp.dirname(self.template_copy_file),
                self.template_copy_file,
                pdf_file_path,
                timeout=self.timeout,
            )
            return {self.output_key[0]: pdf_file_path}

        except Exception as e:
            print(f"Error occurred: {e}")
            return None
