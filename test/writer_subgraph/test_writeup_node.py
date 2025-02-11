import pytest
from unittest.mock import patch, MagicMock
from typing import TypedDict
import json
import re
from researchgraph.writer_subgraph.nodes.writeup_node import WriteupNode, regex_rules


class WriteupState(TypedDict):
    objective: str
    base_method_text: str
    add_method_text: str
    new_method_text: list
    base_method_code: str
    add_method_code: str
    new_method_code: list


@pytest.fixture(scope="function")
def sample_state() -> WriteupState:
    return {
        "objective": "Testing the WriteupNode generation.",
        "base_method_text": "Baseline method description...",
        "add_method_text": "Added method description...",
        "new_method_text": ["New combined method description..."],
        "base_method_code": "def base_method(): pass",
        "add_method_code": "def add_method(): pass",
        "new_method_code": ["def new_method(): pass"],
    }

@pytest.fixture
def writeup_node() -> WriteupNode:
    return WriteupNode(
        llm_name="test-model",
        refine_round=2,
        refine_only=False,
        target_sections=["Title", "Abstract", "Introduction", "Method", "Results"],
    )

def test_writeup_node_generate_note(writeup_node: WriteupNode, sample_state: WriteupState):
    """
    _generate_note() が正しく state をセクションごとにまとめるかをテスト。
    """
    note = writeup_node._generate_note(sample_state)
    for section_name in regex_rules.keys():
        assert f"# {section_name}" in note
    assert "objective: Testing the WriteupNode generation." in note

@patch("researchgraph.writer_subgraph.nodes.writeup_node.completion")
def test_writeup_node_call_llm(mock_completion, writeup_node: WriteupNode):
    """
    _call_llm() が LLM からのレスポンスを正しくパースできるかをテスト。
    """
    mock_completion.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content=json.dumps({"generated_paper_text": "Mock LLM output"})
                )
            )
        ]
    )

    prompt = "Test prompt"
    result = writeup_node._call_llm(prompt)
    assert result == "Mock LLM output"

@pytest.mark.parametrize(
    "invalid_response",
    [
        '{"invalid_key":"No generated_paper_text key"}',
        '{"generated_paper_text":123}',
        '',
    ],
)
@patch("researchgraph.writer_subgraph.nodes.writeup_node.completion")
def test_writeup_node_call_llm_invalid_response(mock_completion, writeup_node, invalid_response):
    """
    LLM が想定外の JSON を返した際のエラー動作をテスト。
    """
    mock_completion.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content=invalid_response
                )
            )
        ]
    )
    with pytest.raises((Exception)):
        writeup_node._call_llm("Test prompt")

def test_writeup_node_clean_meta_information(writeup_node: WriteupNode):
    """
    _clean_meta_information() がメタ情報や指示を取り除けるかをテスト。
    """
    text_with_meta = """
Here is a refined version. 
Certainly! This section discusses ...
% This is a comment
--- 
**Some instructions**:
_Editor@cref
"""
    cleaned_text = writeup_node._clean_meta_information(text_with_meta)
    assert "Here is a refined version." not in cleaned_text
    assert "Certainly! This section discusses" not in cleaned_text
    assert "---" not in cleaned_text
    assert "% This is a comment" not in cleaned_text
    assert "_Editor@cref" not in cleaned_text

def test_writeup_node_generate_write_prompt(writeup_node: WriteupNode):
    """
    _generate_write_prompt() がセクション名とノートを元に正しいテンプレートを生成するかをテスト。
    """
    section = "Method"
    note = "This is a note."
    prompt = writeup_node._generate_write_prompt(section, note)
    assert "You are tasked with filling in the 'Method' section" in prompt
    assert "This is a note." in prompt
    assert "Some tips are provided below:" in prompt

def test_writeup_node_generate_refinement_prompt(writeup_node: WriteupNode):
    """
    _generate_refinement_prompt() がセクション名・ノート・コンテンツを元に正しいテンプレートを生成するかをテスト。
    """
    section = "Method"
    note = "This is a note for refinement."
    content = "This is the original content."
    prompt = writeup_node._generate_refinement_prompt(section, note, content)
    assert "Now criticize and refine only the Method that you just wrote" in prompt
    assert "This is a note for refinement." in prompt
    assert "This is the original content." in prompt
    assert "Some tips are provided below:" in prompt

@patch.object(WriteupNode, "_call_llm", return_value="LLM generated content.")
def test_writeup_node_write(mock_call_llm, writeup_node: WriteupNode):
    """
    _write() が _call_llm() を呼び出し、その戻り値を返すかをテスト。
    """
    note = "Dummy note"
    section = "Title"
    result = writeup_node._write(note, section)
    mock_call_llm.assert_called_once()
    assert result == "LLM generated content."


@patch.object(WriteupNode, "_call_llm", return_value="Refined content.")
def test_writeup_node_refine(mock_call_llm, writeup_node: WriteupNode):
    """
    _refine() が refine_round の回数だけ _call_llm を呼び出し、最終結果を返すかをテスト。
    """
    note = "Dummy note"
    section = "Abstract"
    content = "Initial content"
    writeup_node.refine_round = 3
    result = writeup_node._refine(note, section, content)
    assert mock_call_llm.call_count == 3
    assert result == "Refined content."


@patch("researchgraph.writer_subgraph.nodes.writeup_node.completion")
def test_writeup_node_execute(mock_completion, writeup_node: WriteupNode, sample_state: WriteupState):
    mock_completion.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content=json.dumps({"generated_paper_text": "MOCKED CONTENT"})
                )
            )
        ]
    )

    result_dict = writeup_node.execute(sample_state)

    for section in ["Title", "Abstract", "Introduction", "Method", "Results"]:
        assert section in result_dict
        # ここで "MOCKED CONTENT" が返るはず
        assert result_dict[section] == "MOCKED CONTENT"

def test_writeup_node_execute_refine_only(writeup_node: WriteupNode, sample_state: WriteupState):
    """
    refine_only=True の場合は生成を行わず、既存のセクション内容を refine するだけとなるかをテスト。
    """
    writeup_node.refine_only = True
    custom_state = sample_state.copy()
    custom_state["Title"] = "Existing title content"
    custom_state["Abstract"] = "Existing abstract content"

    with patch.object(
        writeup_node, "_refine", return_value="Refined existing content"
    ) as mock_refine, patch.object(
        writeup_node, "_write"
    ) as mock_write:
        result_dict = writeup_node.execute(custom_state)

    mock_write.assert_not_called()

    assert mock_refine.call_count == len(writeup_node.target_sections)
    assert all(
        val == "Refined existing content" for val in result_dict.values()
    ), "Refine 結果が正しく返されていない"


def test_writeup_node_regex_rules_validation(sample_state: WriteupState):
    """
    regex_rules が想定通りのキーとマッチングしているかをテスト。
    """
    pattern_title = regex_rules.get("Title")
    assert pattern_title is not None
    assert re.search(pattern_title, "objective") is not None, \
        "objective が Title セクションにマッチしない"

    pattern_methods = regex_rules.get("Methods")
    assert pattern_methods is not None
    assert re.search(pattern_methods, "base_method_text") is not None, \
        "base_method_text が Methods セクションにマッチしない"

@patch("researchgraph.writer_subgraph.nodes.writeup_node.completion")
def test_writeup_node_relate_work(mock_completion, writeup_node: WriteupNode):
    """
    _relate_work() が将来的に関連研究を処理する機能を持つ予定だが、現状は素通し実装のため簡単な確認のみ。
    """
    input_text = "Some related work content."
    output_text = writeup_node._relate_work(input_text)
    assert output_text == input_text, "_relate_work() が現状は入力をそのまま返していない"


@patch("researchgraph.writer_subgraph.nodes.writeup_node.completion", side_effect=Exception("Mock LLM error"))
def test_writeup_node_call_llm_exception(mock_completion, writeup_node: WriteupNode):
    """
    LLM への呼び出し（_call_llm）がエラーを投げた場合の挙動をテスト。
    """
    with pytest.raises(Exception) as exc_info:
        _ = writeup_node._call_llm("Prompt that triggers error.")
    assert "Mock LLM error" in str(exc_info.value), "想定外のエラー内容"

def test_writeup_node_init_defaults():
    """
    WriteupNode のデフォルト引数の確認テスト。
    """
    node = WriteupNode(llm_name="default-model")
    assert node.llm_name == "default-model"
    assert node.refine_round == 2
    assert node.refine_only is False
    expected_sections = [
        "Title",
        "Abstract",
        "Introduction",
        "Related work",
        "Background",
        "Method",
        "Experimental setup",
        "Results",
        "Conclusions",
    ]
    assert node.target_sections == expected_sections
