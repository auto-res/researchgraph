import json
import pytest
from researchgraph.latex_subgraph.nodes.convert_to_latex import convert_to_latex


@pytest.fixture
def latex_fake_response() -> dict:
    return {
        "Title": "LaTeX Title",
        "Abstract": "This is abstract.",
        "Introduction": "Intro text",
        "Related_Work": "Related work",
        "Background": "Background",
        "Method": "Method",
        "Experimental_Setup": "Setup",
        "Results": "Results",
        "Conclusions": "Conclusion"
    }


def test_convert_to_latex_success(monkeypatch: pytest.MonkeyPatch, latex_fake_response) -> None:
    monkeypatch.setattr(
        "researchgraph.latex_subgraph.nodes.convert_to_latex.openai_client",
        lambda *args, **kwargs: json.dumps(latex_fake_response)
    )

    result = convert_to_latex(
        llm_name="dummy", 
        prompt_template="", 
        paper_content={key: "dummy" for key in latex_fake_response.keys()}
    )
    assert result["Title"] == "LaTeX Title"


@pytest.mark.parametrize(
    "raw_response, expected_msg",
    [
        (None,                          "No response"),
        ("",                            "No response"),
        ("{}",                          "Missing or empty"),
        ('{"Title": ""}', "Missing or empty"),
    ],
)
def test_convert_to_latex_errors(
    monkeypatch: pytest.MonkeyPatch, 
    latex_fake_response, 
    raw_response, 
    expected_msg
) -> None:
    monkeypatch.setattr(
        "researchgraph.latex_subgraph.nodes.convert_to_latex.openai_client",
        lambda *args, **kwargs: raw_response
    )

    with pytest.raises(ValueError) as exc:
        convert_to_latex(
            llm_name="dummy",
            prompt_template="",
            paper_content={key: "dummy" for key in latex_fake_response.keys()}
        )
    assert expected_msg in str(exc.value)