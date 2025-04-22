import pytest
from researchgraph.html_subgraph.nodes.convert_to_html import convert_to_html


def test_convert_to_html_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "researchgraph.html_subgraph.nodes.convert_to_html.openai_client",
        lambda *args, **kwargs: '{"generated_html_text": "<p>OK</p>"}'
    )

    result = convert_to_html(
        llm_name="dummy", 
        prompt_template="", 
        paper_content={"sec1": "text"}
    )
    assert result == "<p>OK</p>"


@pytest.mark.parametrize(
    "raw_response, expected_msg",
    [
        (None,                          "No response"),
        ("",                            "No response"),
        ("{}",                          "Empty HTML content"),
        ('{"generated_html_text": ""}', "Empty HTML content"),
    ],
)
def test_convert_to_html_errors(monkeypatch: pytest.MonkeyPatch, raw_response, expected_msg) -> None:
    monkeypatch.setattr(
        "researchgraph.html_subgraph.nodes.convert_to_html.openai_client",
        lambda *args, **kwargs: raw_response
    )

    with pytest.raises(ValueError) as exc:
        convert_to_html(
            llm_name="dummy",
            prompt_template="",
            paper_content={"sec1": "text"}
        )
    assert expected_msg in str(exc.value)