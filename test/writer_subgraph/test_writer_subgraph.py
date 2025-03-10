import pytest
from unittest.mock import patch
from researchgraph.writer_subgraph.writer_subgraph import WriterSubgraph, WriterState


@pytest.fixture(scope="function")
def test_state():
    """テスト用の WriterState を作成"""
    return WriterState(
        objective="Mock Objective",
        base_method_text="Base method text",
        add_method_text="Additional method text",
        new_method_text=[],
        base_method_code="Base method code",
        add_method_code="Additional method code",
        new_method_code=[],
        paper_content={},
        tex_text="",
        github_owner="mock_owner",
        repository_name="mock_repo",
        branch_name="mock_branch",
        add_github_url="mock_add_url",
        base_github_url="mock_base_url",
        completion=False,
        devin_url="mock_devin_url"
    )


@pytest.fixture
def mock_nodes():
    """各ノードのモックを作成"""
    mocks = {}
    with patch("researchgraph.writer_subgraph.nodes.writeup_node.WriteupNode.execute", return_value={"Title": "Mock Title", "Abstract": "Mock Abstract"}) as mock_writeup, \
        patch("researchgraph.writer_subgraph.nodes.latexnode.LatexNode.execute", return_value="LaTeX Content") as mock_latex, \
        patch("researchgraph.writer_subgraph.nodes.github_upload_node.GithubUploadNode.execute", return_value=True) as mock_github:

        mocks["writeup_node"] = mock_writeup
        mocks["latex_node"] = mock_latex
        mocks["github_upload_node"] = mock_github
        
        yield mocks


@pytest.fixture
def writer_subgraph():
    """WriterSubgraph のテスト用インスタンスを作成"""
    latex_template_file_path = "/mock/path/to/template.tex"
    figures_dir = "/mock/path/to/figures"
    pdf_file_path = "/mock/path/to/generated.pdf"
    llm_name = "gpt-4o-mini-mock"
    
    return WriterSubgraph(
        llm_name=llm_name,
        latex_template_file_path=latex_template_file_path,
        figures_dir=figures_dir,
        pdf_file_path=pdf_file_path, 
    ).build_graph()


def test_writer_subgraph(mock_nodes, test_state, writer_subgraph):
    """WriterSubgraph の統合テスト"""
    result = writer_subgraph.invoke(test_state)

    for node in ["writeup_node", "latex_node", "github_upload_node"]:
        mock_nodes[node].assert_called_once()

    assert result["completion"] is True
    assert result["tex_text"] == "LaTeX Content"
    assert result["paper_content"]["Title"] == "Mock Title"
    assert result["paper_content"]["Abstract"] == "Mock Abstract"


def test_writeup_node(mock_nodes, test_state, writer_subgraph):
    """LangGraphを通じた WriteupNode の統合テスト"""
    result = writer_subgraph.invoke(test_state)
    mock_nodes["writeup_node"].assert_called_once()

    assert result["paper_content"]["Title"] == "Mock Title"
    assert result["paper_content"]["Abstract"] == "Mock Abstract"


def test_latex_node(mock_nodes, test_state, writer_subgraph):
    """LangGraphを通じた LatexNode の統合テスト"""
    result = writer_subgraph.invoke(test_state)
    mock_nodes["latex_node"].assert_called_once()

    assert result["tex_text"] == "LaTeX Content"


def test_github_upload_node(mock_nodes, test_state, writer_subgraph):
    """LangGraphを通じた GithubUploadNode の統合テスト"""
    test_state["paper_content"] = {"Title": "Mock Title", "Abstract": "Mock Abstract"}

    result = writer_subgraph.invoke(test_state)
    mock_nodes["github_upload_node"].assert_called_once()

    assert result["completion"] is True
