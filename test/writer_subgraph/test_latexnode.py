import os
import pytest
import shutil
import subprocess
from unittest.mock import patch, MagicMock
from typing import TypedDict
from langgraph.graph import StateGraph
from researchgraph.writer_subgraph.nodes.latexnode import LatexNode


class State(TypedDict):
    paper_content: dict
    pdf_file_path: str

@pytest.fixture(scope="function")
def test_environment(tmp_path_factory):
    """テスト用の一時環境を作成"""
    temp_dir = tmp_path_factory.mktemp("latex_tests")
    template_file = temp_dir / "template.tex"    
    template_copy_file = temp_dir / "template_copy.tex"   
    template_text = r"""
\documentclass{article}
\usepackage{graphicx}
\title{TITLE HERE}
\begin{document}
\begin{abstract}
ABSTRACT HERE
\end{abstract}

\begin{filecontents}{references.bib}
@article{sample_ref, author = {Author}, title = {Title}, year = {2023}}
\end{filecontents}

\end{document}
"""
    template_file.write_text(template_text)
    shutil.copyfile(template_file, template_copy_file)

    figures_dir = temp_dir / "images"
    figures_dir.mkdir()
    pdf_file_path = temp_dir / "test_output.pdf"

    return {
        "temp_dir": temp_dir,
        "template_file": template_file,
        "template_copy_file": template_copy_file,
        "figures_dir": figures_dir,
        "pdf_file_path": pdf_file_path,
    }

@pytest.fixture(autouse=True)
def mock_llm_completions():
    with patch("researchgraph.writer_subgraph.nodes.latexnode.completion") as mock_completion:
        mock_completion.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"latex_full_text":"\\\\documentclass{article}\\n\\\\usepackage{graphicx}\\n\\\\begin{document}\\n\\\\includegraphics{figure1.png}\\n\\\\end{document}"}'
                        
                    )
                )
            ]
        )
        yield

@pytest.fixture
def latex_node(test_environment):
    return LatexNode(
        llm_name="gpt-4o-mini-2024-07-18",
        latex_template_file_path=str(test_environment["template_file"]),
        figures_dir=str(test_environment["figures_dir"]),
        timeout=30,
    )

def test_latex_node_execution(latex_node, test_environment):
    """ 正常系テスト: LatexNode が正しく実行され、PDF が生成されるか """
    def latex_node_callable(state):
        return {"pdf_file_path": latex_node.execute(state["paper_content"], state["pdf_file_path"])}

    graph_builder = StateGraph(State)
    graph_builder.add_node("latexnode", latex_node_callable)
    graph_builder.set_entry_point("latexnode")
    graph_builder.set_finish_point("latexnode")
    graph = graph_builder.compile()

    state = {
        "paper_content": {
            "title": "test title",
            "abstract": "Test Abstract.",
            "introduction": "This is the introduction.",
        },
        "pdf_file_path": str(test_environment["pdf_file_path"]),
    }
    result_state = graph.invoke(state, debug=True)

    assert result_state is not None
    assert os.path.exists(test_environment["pdf_file_path"])

@pytest.mark.parametrize("invalid_template_path", [
    "non_existent_file.txt", 
    "", 
])
def test_latex_node_missing_template(invalid_template_path, test_environment):
    """ 異常系テスト: テンプレートが無効な場合 """
    with pytest.raises(FileNotFoundError):
        node = LatexNode(
            llm_name="gpt-4o",
            latex_template_file_path=invalid_template_path,
            figures_dir=str(test_environment["figures_dir"]),
            timeout=30,
        )
        node._copy_template()

def test_latex_node_check_figures(latex_node, test_environment):
    """ _check_figures() が正しく画像をLaTeXに反映できるか """
    with patch("researchgraph.writer_subgraph.nodes.latexnode.completion") as mock_completion:
        mock_completion.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"latex_full_text":"\\\\documentclass{article}\\n\\\\usepackage{graphicx}\\n\\\\begin{document}\\n\\\\includegraphics{figure1.png}\\n\\\\end{document}"}'
                    )
                )
            ]
        )

        figures_dir = test_environment["figures_dir"]
        (figures_dir / "figure1.png").touch()

        tex_text = r"""
        \documentclass{article}
        \usepackage{graphicx}
        \begin{document}
        \includegraphics{figure1.png}
        \end{document}
        """
        result_text = latex_node._check_figures(tex_text, str(figures_dir))
        assert "figure1.png" in result_text, "figure1.png が LaTeX に正しく反映されていない"

def test_latex_node_missing_figures(latex_node, test_environment):
    """ 画像ディレクトリが空、もしくは存在しない画像を参照している場合の挙動を確認 """
    empty_fig_dir = test_environment["temp_dir"] / "empty_images"
    empty_fig_dir.mkdir(exist_ok=True)
    tex_text = r"""
\documentclass{article}
\usepackage{graphicx}
\begin{document}
\title{TITLE HERE}
\abstract{ABSTRACT HERE}
\end{document}
"""
    result_text = latex_node._check_figures(tex_text, str(empty_fig_dir))
    assert result_text == tex_text

def test_latex_node_invalid_content(latex_node):
    """ LLM がエラーを返した場合の挙動をテスト """
    with patch("researchgraph.writer_subgraph.nodes.latexnode.completion") as mock_completion:
        mock_completion.side_effect = Exception("Mocked LLM error")

        with pytest.raises(Exception):
            latex_node._call_llm("")

def test_latex_node_fill_template(latex_node):
    """ _fill_template() が未定義のプレースホルダを含む場合の挙動を確認 """
    content = {
        "title": "Test Title",
        "abstract": "Test Abstract",
        "unknown_section": "This should not be replaced",
    }
    result_text = latex_node._fill_template(content)

    placeholders = ["TITLE HERE", "ABSTRACT HERE"]
    for ph in placeholders:
        assert ph not in result_text

def test_latex_node_check_references(latex_node):
    """ _check_references() が適切に参照をチェックできるか """
    tex_text = r"""
    \documentclass{article}
    \begin{document}
    \cite{valid_ref}
    \begin{filecontents}{references.bib}
    @article{valid_ref, author = {Author}, title = {Title}, year = {2023}}
    \end{filecontents}
    \end{document}
    """
    result_text = latex_node._check_references(tex_text)
    assert "Missing references found" not in result_text

def test_latex_node_missing_references(latex_node):
    """ _check_references() で references.bib が見つからない場合 """
    tex_text = r"""
\documentclass{article}
\begin{document}
\cite{missing_reference}
\end{document}
"""
    with pytest.raises(FileNotFoundError):
        latex_node._check_references(tex_text)

@patch("researchgraph.writer_subgraph.nodes.latexnode.os.popen")
def test_latex_node_fix_latex_no_errors(mock_popen, latex_node, test_environment):
    """ _fix_latex_errors() がエラーなしの場合に元のLaTeXを返すか """
    mock_popen.return_value.read.return_value = ""

    original_tex_text = r"""
    \documentclass{article}
    \begin{document}
    This is a valid LaTeX document.
    \end{document}
    """
    
    with open(test_environment["template_file"], "w") as f:
        f.write(original_tex_text)

    result_text = latex_node._fix_latex_errors(str(test_environment["template_file"]))
    assert result_text == original_tex_text

@patch("researchgraph.writer_subgraph.nodes.latexnode.os.popen")
def test_latex_node_fix_latex_errors(mock_popen, latex_node, test_environment):
    """ _fix_latex_errors() がエラーを修正できるか """
    mock_popen.return_value.read.return_value = "1: Undefined control sequence."

    with patch.object(latex_node, "_call_llm", return_value="Fixed LaTeX text") as mock_llm:
        result_text = latex_node._fix_latex_errors(str(test_environment["template_file"]))
        assert result_text == "Fixed LaTeX text"
        mock_llm.assert_called_once()

@patch("researchgraph.writer_subgraph.nodes.latexnode.subprocess.run")
def test_latex_node_compile_latex(mock_subprocess, latex_node, test_environment):
    """ _compile_latex() で subprocess.run をモックし、例外が発生しないかテスト """
    mock_subprocess.return_value = MagicMock(stdout="Success", stderr="")

    try:
        latex_node._compile_latex(
            cwd=str(test_environment["temp_dir"]),
            template_copy_file=str(test_environment["template_file"]),
            pdf_file_path=str(test_environment["pdf_file_path"]),
            timeout=30,
        )
        assert os.path.exists(test_environment["pdf_file_path"]) is False
    except Exception:
        pytest.fail("Latex compilation failed unexpectedly")

@patch("researchgraph.writer_subgraph.nodes.latexnode.subprocess.run")
def test_latex_node_compile_latex_failure(mock_subprocess, latex_node, test_environment):
    """ _compile_latex() の失敗時の挙動をテスト """
    mock_subprocess.side_effect = subprocess.CalledProcessError(1, "pdflatex")

    try:
        latex_node._compile_latex(
            cwd=str(test_environment["temp_dir"]),
            template_copy_file=str(test_environment["template_file"]),
            pdf_file_path=str(test_environment["pdf_file_path"]),
            timeout=30,
        )
    except subprocess.CalledProcessError:
        pass
    except Exception:
        pytest.fail("予期しない例外が発生")

@patch("researchgraph.writer_subgraph.nodes.latexnode.completion")
def test_latex_node_call_llm_invalid_response(mock_completion, latex_node):
    mock_completion.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content='{"latex_full_text": 123}'
                )
            )
        ]
    )
    with pytest.raises(Exception):
        latex_node._call_llm("Test prompt")
