import json
import pytest
from researchgraph.html_subgraph.nodes.convert_to_html import convert_to_html


def test_convert_to_html_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "researchgraph.html_subgraph.nodes.convert_to_html.openai_client",
        lambda *args, **kwargs: json.dumps({"generated_html_text": "<p>OK</p>"})
    )

    result = convert_to_html(
        llm_name="dummy", 
        prompt_template="", 
        paper_content={"sec1": "text"}
    )
    assert result == "<p>OK</p>"


def test_convert_to_html_raises_on_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "researchgraph.html_subgraph.nodes.convert_to_html.openai_client",
        lambda *args, **kwargs: None
    )

    with pytest.raises(ValueError) as exc:
        convert_to_html(
            llm_name="dummy",
            prompt_template="",
            paper_content={"sec1": "内容A"}
        )
    assert "No response" in str(exc.value)