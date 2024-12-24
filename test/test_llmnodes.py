import json
from typing import TypedDict
from langgraph.graph import StateGraph
from researchgraph.nodes.llmnode.structured_output.structured_llmnode import (
    StructuredLLMNode,
)
from researchgraph.nodes.llmnode.llmlinks.llmlinks_llmnode import LLMLinksLLMNode
from researchgraph.nodes.llmnode.prompts.idea_generation_prompt import idea_generation_prompt
from researchgraph.nodes.llmnode.prompts.code_generation_prompt import code_generation_prompt


class State(TypedDict):
    week: str
    name: str
    date: str
    participants: list[str]
    source: str
    language: str
    translation1: str

    base_method_text: str
    base_method_code: str
    num_ideas: int
    generated_ideas: list

    selected_idea: str                  # TODO: state["generated_ideas"]の"description"を必要に応じて含める
    improved_method_text: str
    improved_method_code: str


def test_structured_llmnode():
    input_key = ["week"]
    output_key = ["name", "date", "participants"]
    llm_name = "gpt-4o-2024-08-06"
    prompt_template = """
    Extract the event information.
    information：Alice and Bob are going to a science fair on {{week}}.
    """
    graph_builder = StateGraph(State)

    graph_builder.add_node(
        "LLMNode",
        StructuredLLMNode(
            input_key=input_key,
            output_key=output_key,
            llm_name=llm_name,
            prompt_template=prompt_template,
        ),
    )
    graph_builder.set_entry_point("LLMNode")
    graph_builder.set_finish_point("LLMNode")
    graph = graph_builder.compile()
    state = {
        "week": "Friday",
    }
    assert graph.invoke(state, debug=True)


def test_llmlinks_llmnode():
    input_key = ["source", "language"]
    output_key = ["translation1"]
    llm_name = "gpt-4o-2024-08-06"
    prompt_template = """
<source>
{source}
</source>
<language>
{language}
</language>
<rule>
sourceタグで与えられた文章を languageで指定された言語に翻訳して translation1タグを用いて出力せよ．
</rule>
"""

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "LLMLinksLLMNode",
        LLMLinksLLMNode(
            input_key=input_key,
            output_key=output_key,
            llm_name=llm_name,
            prompt_template=prompt_template,
        ),
    )
    graph_builder.set_entry_point("LLMLinksLLMNode")
    graph_builder.set_finish_point("LLMLinksLLMNode")
    graph = graph_builder.compile()
    state = {
        "source": "Hello World!!",
        "language": "japanese",
    }
    assert graph.invoke(state, debug=True)


def test_idea_generation_node():
    llm_model_name = "gpt-4o-2024-08-06"
    prompt_template = idea_generation_prompt

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "idea_generation_node",
        StructuredLLMNode(
            input_key=["base_method_text", "base_method_code", "num_ideas"],
            output_key=["generated_ideas"],
            llm_name=llm_model_name,
            prompt_template=prompt_template,
        ),
    )
    graph_builder.set_entry_point("idea_generation_node")
    graph_builder.set_finish_point("idea_generation_node")
    graph = graph_builder.compile()

    state = {
        "base_method_text": "This is the description of Method A.",
        "base_method_code": "def method_a(): pass",
        "num_ideas": 3,
    }

    result = graph.invoke(state, debug=True)
    assert "generated_ideas" in result, "Output 'generated_ideas' is missing."

    try:
        generated_ideas = json.loads(result["generated_ideas"])
    except json.JSONDecodeError as e:
        assert False, f"Failed to decode 'generated_ideas': {e}"
    assert isinstance(generated_ideas, list), "'generated_ideas' should be a list."

    for idea in generated_ideas:
        assert isinstance(idea, dict), "Each generated idea should be a dictionary."
        assert "idea" in idea, "'idea' key is missing in the generated idea."
        assert "description" in idea, "'description' key is missing in the generated idea."
        assert "score" in idea, "'score' key is missing in the generated idea."
        assert isinstance(idea["score"], (int, float)), "'score' should be a number."

    assert len(generated_ideas) >= 3, "The number of generated ideas should be at least 3."


def test_code_generation_node():
    llm_model_name = "gpt-4o-2024-08-06"
    prompt_template = code_generation_prompt

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "code_generation_node",
        StructuredLLMNode(
            input_key=["base_method_text", "base_method_code", "selected_idea"],
            output_key=["improved_method_text", "improved_method_code"],
            llm_name=llm_model_name,
            prompt_template=prompt_template,
        ),
    )
    graph_builder.set_entry_point("code_generation_node")
    graph_builder.set_finish_point("code_generation_node")
    graph = graph_builder.compile()

    state = {
        "base_method_text": "This is the description of Method A.",
        "base_method_code": "def method_a(): pass",
        "selected_idea": "Integrate transfer learning to improve accuracy.",
    }

    result = graph.invoke(state, debug=True)
    print(f"resultaa: {result}")

    assert "improved_method_text" in result, "Output 'improved_method_text' is missing."
    assert isinstance(result["improved_method_text"], str), "'improved_method_text' should be a string."
    assert len(result["improved_method_text"]) > 0, "'improved_method_text' should not be empty."

    assert "improved_method_code" in result, "Output 'improved_method_code' is missing."
    assert isinstance(result["improved_method_code"], str), "'improved_method_code' should be a string."
    assert len(result["improved_method_code"]) > 0, "'improved_method_code' should not be empty."
