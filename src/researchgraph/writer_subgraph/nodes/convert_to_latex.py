import os
import os.path as osp
import re
import subprocess
import shutil
import json
import tempfile
from pydantic import BaseModel
from typing import Optional
from litellm import completion


class LLMOutput(BaseModel):
    latex_full_text: str


class LatexNode:
    def __init__(
        self,
        llm_name: str,
        figures_dir: str,
        pdf_file_path: str,
        save_dir: str,
        timeout: int = 30,
        latex_template_file_path: str = "/workspaces/researchgraph/src/researchgraph/writer_subgraph/latex/template.tex",
    ):
        self.llm_name = llm_name
        self.latex_template_file_path = latex_template_file_path
        self.figures_dir = figures_dir
        self.pdf_file_path = pdf_file_path
        self.save_dir = save_dir
        self.timeout = timeout

        self.template_dir = osp.dirname(self.latex_template_file_path)

        self.latex_save_dir = osp.join(self.save_dir, "latex")
        os.makedirs(self.latex_save_dir, exist_ok=True)
        self.template_copy_file = osp.join(self.latex_save_dir, "template.tex")

    def _call_llm(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        system_prompt = """
        You are a helpful LaTeX rewriting assistant.
        Please respond ONLY in valid JSON format with a single key "latex_full_text" and no other keys.
        Example:
        {
        "latex_full_text": "..."
        }
        The value of "latex_full_text" must contain the complete LaTeX text.
        Do not add extra keys or text outside the JSON.
        """.strip()

        for attempt in range(max_retries):
            try:
                response = completion(
                    model=self.llm_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    response_format=LLMOutput,
                )
                structured_output = json.loads(response.choices[0].message.content)
                return structured_output["latex_full_text"]
            except Exception as e:
                print(f"[Attempt {attempt+1}/{max_retries}] Error calling LLM: {e}")
        print("Exceeded maximum retries for LLM call.")
        return None

    def _copy_template(self):
        try:
            shutil.copytree(self.template_dir, self.latex_save_dir, dirs_exist_ok=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to copy directory {self.template_dir} to {self.latex_save_dir}: {e}"
            )

    def _fill_template(self, content: dict) -> str:
        # Read the copied template, replace placeholders with content, and save the updated file
        tex_text = ""
        with open(self.template_copy_file, "r") as f:
            tex_text = f.read()

        for section, value in content.items():
            placeholder = f"{section.upper()} HERE"
            if placeholder in tex_text:
                tex_text = tex_text.replace(placeholder, value)
                print(f"置換完了: {placeholder}")
            else:
                print(f"プレースホルダーが見つかりませんでした: {placeholder}")

        with open(self.template_copy_file, "w") as f:
            f.write(tex_text)

        with open(self.template_copy_file, "r") as f:
            updated_tex_text = f.read()

        print("更新後の `template_copy.tex` 内容:\n", updated_tex_text)

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
            return tex_text

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
        print("LLMの実行")
        llm_response = self._call_llm(prompt)
        if not llm_response:
            raise RuntimeError(
                f"LLM failed to respond for missing references: {missing_cites}"
            )
        return llm_response

    def _check_figures(
        self,
        tex_text: str,
        pattern: str = r"\\includegraphics.*?{(.*?)}",
    ) -> str:
        # Verify all referenced figures in the LaTeX content exist in the figures directory
        all_figs = [f for f in os.listdir(self.figures_dir) if f.endswith(".png")]
        if not all_figs:
            print("論文生成に使える図がありません")
            return tex_text

        referenced_figs = re.findall(pattern, tex_text)
        fig_to_use = [fig for fig in referenced_figs if fig in all_figs]
        if not fig_to_use:
            print("論文内で利用している図はありません")
            return tex_text

        prompt = f"""
        LaTeX Text:
        {tex_text}
        Available Images:
        {fig_to_use}
        Please modify and output the above Latex text based on the following instructions.
        - Only “Available Images” are available. 
        - If a figure is mentioned on Latex Text, please rewrite the content of Latex Text to cite it.
        - Do not use diagrams that do not exist in “Available Images”.
        - Return the complete LaTeX text."""
        print("LLMの実行")
        llm_response = self._call_llm(prompt)
        # if not llm_response:
        #     raise RuntimeError(f"LLM failed to respond for missing figures: {missing_figs}")
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
                print("LLMの実行")

                llm_response = self._call_llm(prompt)
                if not llm_response:
                    raise RuntimeError(
                        f"LLM failed to respond for missing figures: {duplicates}"
                    )
                tex_text = llm_response
        return tex_text

    def _fix_latex_errors(self, tex_text: str) -> str:
        # Fix LaTeX errors iteratively using chktex and automated suggestions
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".tex", delete=True
            ) as tmp_file:
                tmp_file.write(tex_text)
                tmp_file.flush()

                ignored_warnings = "-n2 -n24 -n13 -n1 -n8 -n29 -n36 -n44"
                check_cmd = f"chktex {tmp_file.name} -q {ignored_warnings}"
                check_output = os.popen(check_cmd).read()

                if check_output:
                    error_messages = check_output.strip().split("\n")
                    formatted_errors = "\n".join(
                        f"- {msg}" for msg in error_messages if msg
                    )
                    print(f"LaTeX エラー検出: {formatted_errors}")

                    prompt = f"""
                    LaTeX text:
                    {tex_text}

                    Please fix the following LaTeX errors: {formatted_errors}.      
                    Make the minimal fix required and do not remove or change any packages unnecessarily.
                    Pay attention to any accidental uses of HTML syntax, e.g. </end instead of \\end.

                    Return the complete corrected LaTeX text.
                    """
                    print("LLMの実行")

                    llm_response = self._call_llm(prompt)
                    if not llm_response:
                        raise RuntimeError("LLM failed to fix LaTeX errors")
                    return llm_response
                else:
                    print("No LaTex errors found by chktex.")
                    return tex_text
        except FileNotFoundError:
            print("chktex command not found. Skipping LaTeX checks.")
            return tex_text

    def _compile_latex(self, cwd: str):
        # Compile the LaTeX document to PDF using pdflatex and bibtex commands
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
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout,
                    check=True,
                )
                print("Standard Output:\n", result.stdout)
                print("Standard Error:\n", result.stderr)
            except subprocess.TimeoutExpired as e:
                print(f"Latex command timed out: {e}")
            except subprocess.CalledProcessError as e:
                print(f"Error running command {' '.join(command)}: {e}")
            except FileNotFoundError:
                print(
                    f"Command not found: {' '.join(command)}. "
                    "Make sure pdflatex and bibtex are installed and on your PATH."
                )
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

        print("FINISHED GENERATING LATEX")
        pdf_filename = f"{osp.splitext(osp.basename(self.template_copy_file))[0]}.pdf"
        try:
            shutil.move(osp.join(cwd, pdf_filename), self.pdf_file_path)
        except FileNotFoundError:
            print("Failed to rename PDF.")

    def execute(self, paper_content: dict) -> str:
        """
        Main entry point:
        1. Copy template
        2. Fill placeholders
        3. Iterate checks (refs, figures, duplicates, minimal error fix)
        4. Compile
        """
        self._copy_template()
        tex_text = self._fill_template(paper_content)
        max_iterations = 5
        iteration_count = 0

        while iteration_count < max_iterations:
            print(f"Start iteration: {iteration_count}")

            print("Check references...")
            original_tex_text = tex_text
            tex_text = self._check_references(tex_text)
            if tex_text != original_tex_text:
                iteration_count += 1
                continue

            print("Check figures...")
            original_tex_text = tex_text
            tex_text = self._check_figures(tex_text)
            if tex_text != original_tex_text:
                iteration_count += 1
                continue

            print("Check duplicates...")
            original_tex_text = tex_text
            tex_text = self._check_duplicates(
                tex_text,
                {
                    "figure": r"\\includegraphics.*?{(.*?)}",
                    "section header": r"\\section{([^}]*)}",
                },
            )
            if tex_text != original_tex_text:
                iteration_count += 1
                continue

            print("Check LaTeX errors...")
            original_tex_text = tex_text
            tex_text = self._fix_latex_errors(tex_text)
            if tex_text != original_tex_text:
                iteration_count += 1
                continue

            print("No changes detected, exiting loop.")
            break

        if iteration_count == max_iterations:
            print(f"Maximum iterations reached ({max_iterations}), exiting loop.")

        with open(self.template_copy_file, "w") as f:
            f.write(tex_text)

        try:
            self._compile_latex(osp.dirname(self.template_copy_file))
            return tex_text

        except Exception as e:
            print(f"Error occurred during compiling: {e}")
            return tex_text


if __name__ == "__main__":
    state = {
        "paper_content": {
            "title": "test title",
            "abstract": "Test Abstract.",
            "introduction": "This is the introduction.",
        },
        "tex_text": "",
    }
    paper_content = state["paper_content"]
    tex_text = state["tex_text"]
    llm_name = "gpt-4o-mini-2024-07-18"
    latex_template_file_path = (
        "/workspaces/researchgraph/src/researchgraph/writer_subgraph/latex/template.tex"
    )
    save_dir = "/workspaces/researchgraph/data"
    figures_dir = "/workspaces/researchgraph/data/images"
    os.makedirs(figures_dir, exist_ok=True)

    pdf_file_path = "/workspaces/researchgraph/data/test_output.pdf"
    tex_text = LatexNode(
        llm_name=llm_name,
        latex_template_file_path=latex_template_file_path,
        save_dir=save_dir,
        figures_dir=figures_dir,
        pdf_file_path=pdf_file_path,
        timeout=30,
    ).execute(paper_content)
