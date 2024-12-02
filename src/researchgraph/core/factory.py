from researchgraph.core.node import Node
from researchgraph.nodes.llmnode.structured_llmnode import StructuredLLMNode
from researchgraph.nodes.llmnode.llmlinks_llmnode import LLMLinksLLMNode
from researchgraph.nodes.writingnode.texnode import LatexNode


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
                return StructuredLLMNode(node_name, **kwargs)
            elif node_name == "llmlinks_llmnode":
                return LLMLinksLLMNode(node_name, **kwargs)
            else:
                raise ValueError(f"Unknown node type: {node_name}")
        elif node_type == "retrievenoe":
            pass
        elif node_type == "writingnode":
            if node_name == "latex_node":
                return LatexNode(node_name, **kwargs)
        elif node_type == "experimentnode":
            pass
        elif node_type == "codingnode":
            pass
        else:
            raise ValueError(f"Unknown node type: {node_type}")
