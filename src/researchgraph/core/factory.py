from researchgraph.core.node import Node
from researchgraph.nodes.llmnode.structured_output.structured_llmnode import (
    StructuredLLMNode,
)
from researchgraph.nodes.llmnode.llmlinks.llmlinks_llmnode import LLMLinksLLMNode
from researchgraph.nodes.writingnode.texnode import LatexNode

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
    def create_node(node_type: str, node_name: str, **kwargs) -> Node:
        """
        Factory method for dynamically generating nodes
        :param node_type: Node type
        :param node_name: Node name
        :param kwargs: Additional arguments when creating a node
        :return: Node instance
        """
        if node_type == "llmnode":
            if node_name == "structured_llmnode":
                return StructuredLLMNode(**kwargs)
            elif node_name == "llmlinks_llmnode":
                return LLMLinksLLMNode(**kwargs)
            else:
                raise ValueError(f"Unknown node type: {node_name}")
        elif node_type == "retrievenoe":
            if node_name == "retrieve_arxiv_text_node":
                return RetrievearXivTextNode(**kwargs)
            elif node_name == "retrieve_github_repository_node":
                return RetrieveGithubRepositoryNode(**kwargs)
        elif node_type == "writingnode":
            if node_name == "latex_node":
                return LatexNode(**kwargs)
        elif node_type == "experimentnode":
            if node_name == "llmsfttrain_node":
                return LLMSFTTrainNode(**kwargs)
            elif node_name == "llminference_node":
                return LLMInferenceNode(**kwargs)
            elif node_name == "llmevaluate_node":
                return LLMEvaluateNode(**kwargs)
        elif node_type == "codingnode":
            pass
        else:
            raise ValueError(f"Unknown node type: {node_type}")
