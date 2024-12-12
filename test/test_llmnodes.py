from typing import TypedDict
from langgraph.graph import StateGraph
from researchgraph.nodes.llmnode.structured_output.structured_llmnode import (
    StructuredLLMNode,
)
from researchgraph.nodes.llmnode.llmlinks.llmlinks_llmnode import LLMLinksLLMNode


class State(TypedDict):
    week: str
    name: str
    date: str
    participants: list[str]
    source: str
    language: str
    translation1: str


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
