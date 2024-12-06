from researchgraph.core.node import Node

from researchgraph.nodes.llmnode.structured_output.structured_llmnode import (
    StructuredLLMNode,
)
from researchgraph.nodes.llmnode.llmlinks.llmlinks_llmnode import LLMLinksLLMNode

from researchgraph.nodes.writingnode import (
    LatexNode,
    Text2ScriptNode,
)
from researchgraph.nodes.experimentnode.llm import (
    LLMSFTTrainNode,
    LLMInferenceNode,
    LLMEvaluateNode,
)

from researchgraph.nodes.retrievenode import (
    RetrievearXivTextNode,
    RetrieveGithubRepositoryNode,
)


class NodeFactory:
    @staticmethod
    def create_node(node_name: str, **kwargs) -> Node:
        """
        Factory method for dynamically generating nodes
        :param node_name: Node name
        :param kwargs: Additional arguments when creating a node
        :return: Node instance
        """
        # LLMnode
        if node_name == "structuredoutput_llmnode":
            return StructuredLLMNode(**kwargs)
        elif node_name == "llmlinks_llmnode":
            return LLMLinksLLMNode(**kwargs)

        # RetrieveNode
        elif node_name == "retrieve_arxiv_text_node":
            return RetrievearXivTextNode(**kwargs)
        elif node_name == "retrieve_github_repository_node":
            return RetrieveGithubRepositoryNode(**kwargs)

        # WritingNode
        elif node_name == "text2script_node":
            return Text2ScriptNode(**kwargs)
        elif node_name == "latex_node":
            return LatexNode(**kwargs)

        # ExperimentNode
        elif node_name == "llmsfttrain_node":
            return LLMSFTTrainNode(**kwargs)
        elif node_name == "llminference_node":
            return LLMInferenceNode(**kwargs)
        elif node_name == "llmevaluate_node":
            return LLMEvaluateNode(**kwargs)

        else:
            raise ValueError(f"Unknown node type: {node_name}")
