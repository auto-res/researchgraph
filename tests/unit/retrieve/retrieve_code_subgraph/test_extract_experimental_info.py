import pytest
import airas.retrieve_code_subgraph.node.extract_experimental_info as mod
from airas.retrieve_code_subgraph.node.extract_experimental_info import (
    extract_experimental_info,
)


# Normal case: vertexai_client returns expected output
def test_extract_experimental_info_success(monkeypatch):
    def fake_vertexai_client(model_name, message, data_model):
        return {"extract_code": "print('code')", "extract_info": "info text"}

    monkeypatch.setattr(mod, "vertexai_client", fake_vertexai_client)
    code, info = extract_experimental_info(
        model_name="gemini-2.0-flash-001",
        method_text="test method",
        repository_content_str="test repo content",
    )
    assert code == "print('code')"
    assert info == "info text"


# Error case: vertexai_client returns None (should raise RuntimeError)
@pytest.mark.parametrize(
    "vertexai_return, expected_exception, expected_msg",
    [
        (None, RuntimeError, "Failed to get response from Vertex AI."),
    ],
)
def test_extract_experimental_info_error(
    monkeypatch, vertexai_return, expected_exception, expected_msg
):
    monkeypatch.setattr(mod, "vertexai_client", lambda *a, **kw: vertexai_return)
    with pytest.raises(expected_exception) as exc:
        extract_experimental_info(
            model_name="gemini-2.0-flash-001",
            method_text="test method",
            repository_content_str="test repo content",
        )
    assert expected_msg in str(exc.value)
