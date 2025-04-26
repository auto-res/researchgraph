import json
import pytest
import airas.latex_subgraph.nodes.convert_to_latex as mod
from airas.latex_subgraph.nodes.convert_to_latex import convert_to_latex


@pytest.fixture
def fake_llm_response() -> dict[str, str]:
    return {
        "Title": "LaTeX Title",
        "Abstract": "This is abstract.",
        "Introduction": "Intro text",
        "Related_Work": "Related work",
        "Background": "Background",
        "Method": "Method",
        "Experimental_Setup": "Setup",
        "Results": "Results",
        "Conclusions": "Conclusion",
    }


def test_convert_to_latex_success(
    monkeypatch: pytest.MonkeyPatch, fake_llm_response: dict[str, str]
) -> None:
    monkeypatch.setattr(
        mod, "openai_client", lambda *args, **kwargs: json.dumps(fake_llm_response)
    )

    result = convert_to_latex(
        llm_name="dummy",
        prompt_template="",
        paper_content={key: "dummy" for key in fake_llm_response.keys()},
    )
    assert result["Title"] == "LaTeX Title"


@pytest.mark.parametrize(
    "raw_response, expected_msg",
    [
        (None, "No response"),
        ("", "No response"),
        ("{}", "Missing or empty"),
        ('{"Title": ""}', "Missing or empty"),
    ],
)
def test_convert_to_latex_errors(
    monkeypatch: pytest.MonkeyPatch,
    fake_llm_response: dict[str, str],
    raw_response: str | None,
    expected_msg: str,
) -> None:
    monkeypatch.setattr(mod, "openai_client", lambda *args, **kwargs: raw_response)

    with pytest.raises(ValueError) as exc:
        convert_to_latex(
            llm_name="dummy",
            prompt_template="",
            paper_content={key: "dummy" for key in fake_llm_response.keys()},
        )
    assert expected_msg in str(exc.value)
