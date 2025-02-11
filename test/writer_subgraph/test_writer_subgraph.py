import os
import pytest
from unittest.mock import patch, MagicMock
from researchgraph.writer_subgraph.writer_subgraph import WriterSubgraph
from researchgraph.writer_subgraph.input_data import writer_subgraph_input_data


@pytest.fixture
def mock_writeup_node():
    """WriteupNode のモック"""
    with patch("researchgraph.writer_subgraph.nodes.writeup_node.WriteupNode.execute") as mock_writeup:
        mock_writeup.return_value = {
            "Title": "Test Paper",
            "Abstract": "This is a test abstract.",
            "Content": "This is test content."
        }
        yield mock_writeup

@pytest.fixture
def mock_latex_node():
    """LatexNode のモック"""
    with patch("researchgraph.writer_subgraph.nodes.latexnode.LatexNode.execute") as mock_latex:
        mock_latex.return_value = "/tmp/test_output.pdf"
        yield mock_latex

@pytest.fixture
def mock_github_upload_node():
    """GithubUploadNode のモック"""
    with patch("researchgraph.writer_subgraph.nodes.github_upload_node.GithubUploadNode.execute") as mock_github:
        mock_github.return_value = True
        yield mock_github

@pytest.fixture
def writer_subgraph():
    """テスト用の WriterSubgraph インスタンス"""
    return WriterSubgraph(
        llm_name="gpt-4o-mini-2024-07-18",
        latex_template_file_path="/tmp/template.tex",
        figures_dir="/tmp/images"
    ).build_graph()

def test_writer_subgraph_success(
    writer_subgraph, mock_writeup_node, mock_latex_node, mock_github_upload_node
):
    """正常系: 一連の処理が正しく行われるか"""
    input_data = writer_subgraph_input_data
    result = writer_subgraph.invoke(input_data)

    assert result is not None
    assert "completion" in result
    assert result["completion"] is True

def test_writer_subgraph_writeup_fail(writer_subgraph):
    """異常系: WriteupNode でエラーが発生した場合"""
    with patch("researchgraph.writer_subgraph.nodes.writeup_node.WriteupNode.execute") as mock_writeup:
        mock_writeup.side_effect = Exception("WriteupNode Error")
        input_data = writer_subgraph_input_data

        with pytest.raises(Exception, match="WriteupNode Error"):
            writer_subgraph.invoke(input_data)

def test_writer_subgraph_latex_fail(writer_subgraph, mock_writeup_node):
    """異常系: LatexNode でエラーが発生した場合"""
    with patch("researchgraph.writer_subgraph.nodes.latexnode.LatexNode.execute") as mock_latex:
        mock_latex.side_effect = Exception("LatexNode Error")
        input_data = writer_subgraph_input_data

        with pytest.raises(Exception, match="LatexNode Error"):
            writer_subgraph.invoke(input_data)

def test_writer_subgraph_github_fail(writer_subgraph, mock_writeup_node, mock_latex_node):
    """異常系: GithubUploadNode でアップロードエラーが発生した場合"""
    with patch("researchgraph.writer_subgraph.nodes.github_upload_node.GithubUploadNode.execute") as mock_github:
        mock_github.side_effect = Exception("GithubUploadNode Error")
        input_data = writer_subgraph_input_data

        with pytest.raises(Exception, match="GithubUploadNode Error"):
            writer_subgraph.invoke(input_data)
