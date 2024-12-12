from typing import List, Dict, Any, Optional
from researchgraph.graphs.ai_scientist.ai_scientist_node.llm import (
    get_response_from_llm,
    extract_json_between_markers,
)

class DraftImprovementComponent:
    def __init__(self, model: str = "gpt-4"):
        self.model = model

    def improve_draft(self, draft: str, feedback: str) -> str:
        """
        Improve a research paper draft based on feedback.

        Args:
            draft (str): The current draft of the paper
            feedback (str): Feedback to incorporate into the draft

        Returns:
            str: The improved draft
        """
        system_message = """You are a scientific writing expert. Your task is to improve a research paper draft based on provided feedback.
        Focus on clarity, scientific rigor, and addressing the specific points in the feedback.
        Maintain the paper's original structure while enhancing its content and presentation."""

        user_message = f"""Please improve the following research paper draft based on the provided feedback.

Draft:
{draft}

Feedback to address:
{feedback}

Please provide the improved version of the draft while maintaining its scientific integrity and addressing all feedback points."""

        improved_draft = get_response_from_llm(
            system_message=system_message,
            user_message=user_message,
            model=self.model
        )

        return improved_draft
