import os
import json
import pytest
import shutil
import subprocess
from unittest.mock import patch, MagicMock
from researchgraph.writer_subgraph.nodes.convert_to_latex import LatexNode
from requests.exceptions import HTTPError


@pytest.fixture(scope="function")
def test_environment(tmp_path_factory):
    """テスト用の一時環境を作成"""
    temp_dir = tmp_path_factory.mktemp("latex_tests")
    template_file = temp_dir / "template.tex"
    template_copy_file = temp_dir / "template_copy.tex"
    figures_dir = temp_dir / "images"
    figures_dir.mkdir()
    pdf_file_path = temp_dir / "test_output.pdf"

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

    return {
        "temp_dir": temp_dir,
        "template_file": template_file,
        "template_copy_file": template_copy_file,
        "figures_dir": figures_dir,
        "pdf_file_path": pdf_file_path,
        "paper_content": {
            "Title": "Test Title",
            "Abstract": "Test Abstract",
            "Introduction": "This is the introduction.",
            "Method": "Mocked method description.",
            "Results": "Mocked results section.",
        },
    }


@pytest.fixture(autouse=True)
def mock_llm_completions():
    """LLM のモックレスポンス"""
    with patch(
        "researchgraph.writer_subgraph.nodes.latexnode.completion"
    ) as mock_completion:
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
    """LatexNode のインスタンスを作成"""
    return LatexNode(
        llm_name="gpt-4o-mini-2024-07-18",
        latex_template_file_path=str(test_environment["template_file"]),
        figures_dir=str(test_environment["figures_dir"]),
        pdf_file_path=str(test_environment["pdf_file_path"]),
        timeout=30,
    )


@pytest.mark.parametrize("invalid_template_path", ["non_existent_file.txt", ""])
def test_missing_template(invalid_template_path, test_environment):
    """異常系テスト: テンプレートが無効な場合"""
    with pytest.raises(FileNotFoundError):
        node = LatexNode(
            llm_name="gpt-4o",
            latex_template_file_path=invalid_template_path,
            figures_dir=str(test_environment["figures_dir"]),
            pdf_file_path=str(test_environment["pdf_file_path"]),
            timeout=30,
        )
        node._copy_template()


def test_check_figures(latex_node, test_environment):
    """_check_figures() が画像をLaTeXに反映できるか"""
    figures_dir = test_environment["figures_dir"]
    (figures_dir / "figure1.png").touch()

    tex_text = r"""
    \documentclass{article}
    \usepackage{graphicx}
    \begin{document}
    \includegraphics{figure1.png}
    \end{document}
    """
    result_text = latex_node._check_figures(tex_text)
    assert "figure1.png" in result_text, "figure1.png が LaTeX に正しく反映されていない"


def test_missing_figures(latex_node, test_environment):
    """存在しない画像を参照している場合の挙動"""
    for f in os.listdir(test_environment["figures_dir"]):
        os.remove(os.path.join(test_environment["figures_dir"], f))
    tex_text = r"""
\documentclass{article}
\usepackage{graphicx}
\begin{document}
\title{TITLE HERE}
\abstract{ABSTRACT HERE}
\end{document}
"""
    result_text = latex_node._check_figures(tex_text)
    assert result_text == tex_text


def test_invalid_content(latex_node):
    """LLM がエラーを返した場合の挙動"""
    with patch(
        "researchgraph.writer_subgraph.nodes.latexnode.completion",
        side_effect=Exception("Mocked LLM error"),
    ):
        assert latex_node._call_llm("") is None, "LLMのエラー時はNoneを返すべき"


def test_fill_template(latex_node):
    """_fill_template() が未定義のプレースホルダを含む場合の挙動"""
    content = {
        "title": "Test Title",
        "abstract": "Test Abstract",
        "unknown_section": "This should not be replaced",
    }
    result_text = latex_node._fill_template(content)

    placeholders = ["TITLE HERE", "ABSTRACT HERE"]
    for ph in placeholders:
        assert ph not in result_text


def test_check_references(latex_node):
    """_check_references() が適切に参照をチェックするか"""
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


def test_missing_references(latex_node):
    """_check_references() で references.bib が見つからない場合"""
    tex_text = r"""
\documentclass{article}
\begin{document}
\cite{missing_reference}
\end{document}
"""
    with pytest.raises(FileNotFoundError):
        latex_node._check_references(tex_text)


@patch("researchgraph.writer_subgraph.nodes.latexnode.os.popen")
def test_fix_latex_no_errors(mock_popen, latex_node, test_environment):
    """_fix_latex_errors() がエラーなしの場合に元のLaTeXを返すか"""
    mock_popen.return_value.read.return_value = ""

    original_tex_text = test_environment["template_file"].read_text()
    result_text = latex_node._fix_latex_errors(original_tex_text)
    assert result_text == original_tex_text


@patch("researchgraph.writer_subgraph.nodes.latexnode.os.popen")
def test_fix_latex_errors(mock_popen, latex_node, test_environment):
    """_fix_latex_errors() がエラーを修正できるか"""
    mock_popen.return_value.read.return_value = "1: Undefined control sequence."

    with patch.object(
        latex_node, "_call_llm", return_value="Fixed LaTeX text"
    ) as mock_llm:
        tex_text_with_error = (
            test_environment["template_file"].read_text() + r"\undefinedcommand"
        )
        result_text = latex_node._fix_latex_errors(tex_text_with_error)
        assert result_text == "Fixed LaTeX text"
        mock_llm.assert_called_once()


@patch("researchgraph.writer_subgraph.nodes.latexnode.subprocess.run")
def test_compile_latex(mock_subprocess, latex_node, test_environment):
    """_compile_latex() で subprocess.run をモックし、例外が発生しないかテスト"""
    mock_subprocess.return_value = MagicMock(stdout="Success", stderr="")

    try:
        latex_node._compile_latex(cwd=str(test_environment["temp_dir"]))
        assert os.path.exists(test_environment["pdf_file_path"]) is False
    except Exception:
        pytest.fail("Latex compilation failed unexpectedly")


@patch("researchgraph.writer_subgraph.nodes.latexnode.subprocess.run")
def test_compile_latex_failure(mock_subprocess, latex_node, test_environment):
    """_compile_latex() の失敗時の挙動をテスト"""
    mock_subprocess.side_effect = subprocess.CalledProcessError(1, "pdflatex")

    try:
        latex_node._compile_latex(cwd=str(test_environment["temp_dir"]))
    except subprocess.CalledProcessError:
        pass
    except Exception:
        pytest.fail("予期しない例外が発生")


@patch("researchgraph.writer_subgraph.nodes.latexnode.completion")
def test_execute(mock_completion, latex_node, test_environment):
    mock_completion.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content=json.dumps({"latex_full_text": "MOCKED CONTENT"})
                )
            )
        ]
    )
    result_tex_text = latex_node.execute(test_environment["paper_content"])
    # 置換された内容が含まれていることを確認
    assert "TITLE HERE" not in result_tex_text, "TITLE HERE が残っている"
    assert "ABSTRACT HERE" not in result_tex_text, "ABSTRACT HERE が残っている"
    assert "Test Title" in result_tex_text, "タイトルが正しく置換されていない"
    assert "Test Abstract" in result_tex_text, "アブストラクトが正しく置換されていない"
    assert result_tex_text is not None, "execute() の出力が None になっている"


@pytest.mark.parametrize(
    "exception, expected_message",
    [
        (
            ConnectionError("Mocked Connection Error"),
            "ConnectionError が発生した場合 None を返すべき",
        ),
        (
            TimeoutError("Mocked Timeout Error"),
            "TimeoutError が発生した場合 None を返すべき",
        ),
        (
            HTTPError("Mocked Rate Limit Error (429)"),
            "RateLimitError が発生した場合 None を返すべき",
        ),
        (
            HTTPError("Mocked Internal Server Error (500)"),
            "HTTPError が発生した場合 None を返すべき",
        ),
    ],
)
@patch("researchgraph.writer_subgraph.nodes.latexnode.completion")
def test_call_llm_api_errors(mock_completion, latex_node, exception, expected_message):
    """LLM API 呼び出し時に各種エラーが発生した場合のハンドリング"""
    mock_completion.side_effect = exception

    result = latex_node._call_llm("Test prompt")
    assert result is None, expected_message


@pytest.mark.parametrize(
    "mock_response, expected_message",
    [
        ("INVALID JSON", "JSONDecodeError が発生した場合 None を返すべき"),
        (
            json.dumps({"wrong_key": "Some text"}),
            "KeyError が発生した場合 None を返すべき",
        ),
        (None, "AttributeError が発生した場合 None を返すべき"),
    ],
)
@patch("researchgraph.writer_subgraph.nodes.latexnode.completion")
def test_call_llm_response_errors(
    mock_completion, latex_node, mock_response, expected_message
):
    """LLM のレスポンスが異常だった場合のハンドリング"""
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=mock_response))]
    )

    result = latex_node._call_llm("Test prompt")
    assert result is None, expected_message
