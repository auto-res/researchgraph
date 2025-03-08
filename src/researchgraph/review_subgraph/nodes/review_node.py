import os
import os.path as osp
import time
import json
from pydantic import BaseModel, create_model
from litellm import completion
from jinja2 import Environment
from typing import Optional


# 各サブグラフの出力フィールドを定義
SUBGRAPH_OUTPUT_FIELDS = {
    "retrieve_paper_subgraph": {},
    "generate_subgraph": {},
    "executor_subgraph": {},
    "writer_subgraph": {
        "review_feedback": (str, ...),
        "review_score_clarity": (float, ...),
        "review_score_structure": (float, ...),
        "review_score_technical_depth": (float, ...),
    },
}

class DynamicModel(BaseModel):
    pass


class ReviewNode:
    def __init__(
        self,
        llm_name: str, 
        save_dir: str,
        review_target: str,
        threshold: float = 3.5,
    ):
        self.llm_name = llm_name
        self.save_dir = save_dir
        self.review_target = review_target
        self.threshold = threshold
        self.env = Environment()
        self.dynamic_model = self._create_dynamic_model(DynamicModel)

        self.review_system_prompt = """
        You are an academic peer reviewer. 
        Your task is to evaluate research papers critically, identifying strengths and weaknesses, and suggesting improvements. 
        Your feedback should be constructive, specific, and actionable.

        **Instructions:**
        1. Rate each evaluation criterion on a scale of 1.0 to 5.0, with increments of 0.1:
            - 1.0～2.0 = Needs significant improvement
            - 2.1～3.0 = Below average
            - 3.1～4.0 = Acceptable with revisions
            - 4.1～5.0 = High quality

        **Be conservative in awarding high scores**  
        For a 5.0 rating, the paper or idea should demonstrate exceptional clarity, originality, methodology, and significance with virtually no major flaws.

        2. For each criterion you score:
            - **Provide specific evidence** from the text or proposal to justify the score. 
            - Clearly explain how that evidence supports the numeric rating. 
            - Include **direct references (e.g., “In Section 3, the method is vaguely described...”)** or excerpts to pinpoint strong or weak points.
    
        3. **Identify at least three specific areas for improvement** and provide **clear revision suggestions in the `review_feedback` field**:
            - **What aspects contributed to this score?**
            - **What needs to be changed?**
            - **How should it be revised?**
            - **Why will this improve the paper?**

        **Output Format:**
        Your response must be in the following JSON format:
        { 
        "review_score_xxx": <1.0-5.0>, 
        "review_score_yyy": <1.0-5.0>, 
        "review_score_zzz": <1.0-5.0>, 
        "review_feedback": "<Your feedback here>" 
        }   
        """

        self.review_prompt_dict = {
            "retrieve_paper_subgraph": """
            """,
            
            "generate_subgraph": """
            """,
            
            "executor_subgraph": """
            """,
            
            "writer_subgraph": """
            The following is a research paper written in LaTeX format. Evaluate its quality based on the following criteria:

            **Evaluation Criteria:**
            - **review_score_clarity (1.0-5.0)**: Is the writing clear and understandable?
            - **review_score_structure (1.0-5.0)**: Is the structure and organization of the paper logical and coherent?
            - **review_score_technical_depth (1.0-5.0)**: Does the paper provide sufficient technical depth and explanation?
            - **review_feedback**: Provide feedback on the strengths and weaknesses of the paper and suggest improvements.

            **Research Paper Content (LaTeX format):**
            {{ content }}
            """
        }

    def _create_dynamic_model(self, base_model: BaseModel) -> BaseModel:
        """ review_target に応じた動的な Pydantic モデルを生成する """
        fields = SUBGRAPH_OUTPUT_FIELDS.get(self.review_target, {})
        return create_model(
            f"LLMOutput_{self.review_target.capitalize()}",
            **fields,
            __base__=base_model,
        )

    def _review(self, content: str) -> Optional[tuple[str, dict]]:

        if self.review_target not in self.review_prompt_dict:
            raise ValueError(f"Invalid review target: {self.review_target}")

        template = self.env.from_string(self.review_prompt_dict[self.review_target])
        prompt = template.render({"content": content})
        max_retries = 3
        for attempt in range(max_retries): 
            try:
                response = completion(
                    model=self.llm_name,
                    messages=[
                        {"role": "system", "content": self.review_system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    response_format=self.dynamic_model,
                )
                structured_output = json.loads(response.choices[0].message.content)
                review_feedback = structured_output.get("review_feedback", "")
                review_scores = {
                    key: value for key, value in structured_output.items()
                    if key.startswith("review_score_") and isinstance(value, float)
                }
                return review_feedback, review_scores

            except Exception as e:
                print(f"[Attempt {attempt+1}/{max_retries}] Unexpected error: {e}")
        print("Exceeded maximum retries for LLM call.")
        return None

    def _review_judgement(self, review_scores: dict) -> bool:
        return all(score >= self.threshold for score in review_scores.values())
    
    def _save_review_log(self, content: str, review_decision: bool, review_scores: dict, review_feedback: str): 
        #TODO: 同じワークフロー内で生成されるレビューを1つのファイルに集約できるようにする

        os.makedirs(self.save_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        review_file = osp.join(self.save_dir, f"{timestamp}_review.json")
        review_data = {
            "timestamp": timestamp,
            "review_target": self.review_target,
            "review_decision": "Accepted" if review_decision else "Needs Improvement",
            "review_scores": review_scores,
            "review_feedback": review_feedback,
            "content": content, 
        }
        try:
            with open(review_file, "w") as f:
                json.dump(review_data, f, indent=4)
            print(f"Review log saved to {review_file}")
        except Exception as e:
            print(f"Failed to save review log: {e}")

    def execute(self, content: str) -> tuple[bool, str]:
        print("Reviewing content...")
        review_result = self._review(content)
        if not review_result:
            raise RuntimeError("Failed to review content. The LLM returned an empty response.")

        review_feedback, review_scores = review_result
        review_decision = self._review_judgement(review_scores)
        self._save_review_log(content, review_decision, review_scores, review_feedback)
        
        return review_decision, review_feedback