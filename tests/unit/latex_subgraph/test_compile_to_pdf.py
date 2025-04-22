import os
import pytest
import researchgraph.latex_subgraph.nodes.compile_to_pdf as mod
from typing import Any, Generator
from researchgraph.latex_subgraph.nodes.compile_to_pdf import LatexNode


@pytest.fixture
def tmp_env(tmp_path):
    template_dir = tmp_path / "templatex_dir"
    template_dir.mkdir()
    (template_dir / "template.tex").write_text(
        "TITLE HERE\nABSTRACT HERE\n\\cite{ref1}\n"
    )
    (template_dir / "references.bib").write_text(
        "@article{ref1, author={A}, title={T}, year={2025}}"
    )

    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()

    pdf_file_path = tmp_path / "test.pdf"

    save_dir = tmp_path / "save"
    save_dir.mkdir()

    return {
        "template_dir":  str(template_dir),
        "template_file": str(template_dir / "template.tex"),
        "save_dir":      str(save_dir),
        "figures_dir":   str(figures_dir),
        "pdf_file_path": str(pdf_file_path), 
    }


@pytest.fixture(autouse=True)
def patch_openai(monkeypatch: pytest.MonkeyPatch) -> Generator[None, Any, None]:
    monkeypatch.setattr(
        mod,
        "openai_client", 
        lambda *args, **kwargs: '{"latex_full_text": "DUMMY"}'
    )
    yield


@pytest.fixture
def node(tmp_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> LatexNode:
    """LatexNode のインスタンスを作成"""
    node = LatexNode(
        llm_name="test-model",
        figures_dir=tmp_env["figures_dir"],
        pdf_file_path=tmp_env["pdf_file_path"],
        save_dir=tmp_env["save_dir"],
        timeout=5,
        latex_template_file_path=tmp_env["template_file"],
    )
    monkeypatch.setattr(node, "template_dir", tmp_env["template_dir"])
    return node


def test_copy_template_creates_files(node: LatexNode, tmp_env: dict[str, str]) -> None:
    node._copy_template()
    copied = os.listdir(os.path.join(tmp_env["save_dir"], "latex"))
    assert "template.tex" in copied
    assert "references.bib" in copied


@pytest.mark.parametrize("content, expected", [
    ({"title":"T1", "abstract":"A1"}, ("T1", "A1")),
    ({}, ("TITLE HERE", "ABSTRACT HERE")),
])
def test_fill_template(node: LatexNode, content: dict[str, str], expected: tuple[str, str]) -> None:
    node._copy_template()
    text = node._fill_template(content)
    assert expected[0] in text
    assert expected[1] in text


def test_call_llm_success(node):
    res = node._call_llm("prompt")
    assert res == "DUMMY"


@pytest.mark.parametrize(
    "raw_response, expected_msg",
    [
        (None,                          "No response"),
        ("",                            "No response"),
        ("{}",                          "Empty LaTeX content"),
        ('{"latex_full_text": ""}', "Empty LaTeX content"),
    ],
)
def test_call_llm_errors(node: LatexNode, monkeypatch: pytest.MonkeyPatch, raw_response, expected_msg):
    monkeypatch.setattr(
        mod, 
        "openai_client",
        lambda *args, **kwargs: raw_response
    )
    with pytest.raises(ValueError, match=expected_msg):
        node._call_llm("prompt")

# TODO: カバレッジを上げる