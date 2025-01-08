import json
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from researchgraph.nodes.llmnode.structured_output.structured_llmnode import (
    StructuredLLMNode,
)
from researchgraph.nodes.llmnode.llmlinks.llmlinks_llmnode import LLMLinksLLMNode
from researchgraph.graphs.ai_integrator.ai_integrator_v3.refiner_subgraph.llmnode_prompt import (
    ai_integrator_v3_llmcreator_prompt, 
    ai_integrator_v3_llmcoder_prompt, 
)

class State(BaseModel):
    week: str = Field(default="")
    name: str = Field(default="")
    date: str = Field(default="")
    participants: list[str] = Field(default_factory=list)
    source: str = Field(default="")
    language: str = Field(default="")
    translation1: str = Field(default="")

    base_method_text: str = Field(default="")
    base_method_code: str = Field(default="")
    num_ideas: int = Field(default=1)
    generated_ideas: str = Field(default="")

    refined_method_text: str = Field(default="")
    refined_method_code: str = Field(default="")


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


# def test_llmlinks_llmnode():
#     input_key = ["source", "language"]
#     output_key = ["translation1"]
#     llm_name = "gpt-4o-2024-08-06"
#     prompt_template = """
# <source>
# {source}
# </source>
# <language>
# {language}
# </language>
# <rule>
# sourceタグで与えられた文章を languageで指定された言語に翻訳して translation1タグを用いて出力せよ．
# </rule>
# """

#     graph_builder = StateGraph(State)
#     graph_builder.add_node(
#         "LLMLinksLLMNode",
#         LLMLinksLLMNode(
#             input_key=input_key,
#             output_key=output_key,
#             llm_name=llm_name,
#             prompt_template=prompt_template,
#         ),
#     )
#     graph_builder.set_entry_point("LLMLinksLLMNode")
#     graph_builder.set_finish_point("LLMLinksLLMNode")
#     graph = graph_builder.compile()
#     state = {
#         "source": "Hello World!!",
#         "language": "japanese",
#     }
#     assert graph.invoke(state, debug=True)

def test_llmcreator():
    llm_model_name = "gpt-4o-2024-08-06"
    prompt_template = ai_integrator_v3_llmcreator_prompt

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "llmcreator",
        StructuredLLMNode(
            input_key=["base_method_text", "base_method_code", "num_ideas"],
            output_key=["generated_ideas"],
            llm_name=llm_model_name,
            prompt_template=prompt_template,
        ),
    )
    graph_builder.set_entry_point("llmcreator")
    graph_builder.set_finish_point("llmcreator")
    graph = graph_builder.compile()

    state = {
        "base_method_text": "This is the description of Method A.",
        "base_method_code": "def method_a(): pass",
    }

    result = graph.invoke(state, debug=True)
    assert "generated_ideas" in result, "Output 'generated_ideas' is missing."
    assert isinstance(result["generated_ideas"], str), "'generated_ideas' should be a str."

def test_llmcoder():
    llm_model_name = "gpt-4o-2024-08-06"
    prompt_template = ai_integrator_v3_llmcoder_prompt

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "llmcoder",
        StructuredLLMNode(
            input_key=["base_method_text", "base_method_code", "generated_ideas"],
            output_key=["refined_method_text", "refined_method_code"],
            llm_name=llm_model_name,
            prompt_template=prompt_template,
        ),
    )
    graph_builder.set_entry_point("llmcoder")
    graph_builder.set_finish_point("llmcoder")
    graph = graph_builder.compile()

    state = {
        "base_method_text": "This is the description of Method A.",
        "base_method_code": "def method_a(): pass",
        "generatedd_ideas": "Integrate transfer learning to improve accuracy.",
    }

    result = graph.invoke(state, debug=True)

    assert "refined_method_text" in result, "Output 'refined_method_text' is missing."
    assert isinstance(result["refined_method_text"], str), "'refined_method_text' should be a string."
    assert len(result["refined_method_text"]) > 0, "'refined_method_text' should not be empty."

    assert "refined_method_code" in result, "Output 'refined_method_code' is missing."
    assert isinstance(result["refined_method_code"], str), "'refined_method_code' should be a string."
    assert len(result["refined_method_code"]) > 0, "'refined_method_code' should not be empty."
