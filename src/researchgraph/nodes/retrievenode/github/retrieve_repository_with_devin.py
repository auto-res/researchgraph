import os
import requests
import time
from pydantic import BaseModel, Field
from researchgraph.core.node import Node

API_KEY = os.getenv("DEVIN_API_KEY")

class State(BaseModel):
    devin_prompts: str = Field(default="")
    github_url: str = Field(default="")
    add_method_text: str = Field(default="")
    extracted_code: str = Field(default="")


class RetrieveCodeWithDevinNode(Node):
    def __init__(
        self, input_key: list[str], output_key: list[str]
    ):
        super().__init__(input_key, output_key)
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
            }
        
    def _create_session(self, devin_prompt, github_url, add_method_text):
        url = "https://api.devin.ai/v1/sessions"
        data = {
            "prompt": f"""
            {devin_prompt}
            # Description of Methodology
            {add_method_text}
            # GitHub Repository url
            {github_url}""",
            "idempotent": True
        }
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 200:
            print("Success:", response.json())
        else:
            print("Failed:", response.status_code, response.text)
        session_data = response.json()
        session_id = session_data["session_id"]
        return session_id
        
    def _get_devin_response(self, session_id):
        get_url = f"https://api.devin.ai/v1/session/{session_id}"
        backoff = 1
        max_attempts = 20
        attempts = 0
        while attempts < max_attempts:
            print(f"Attempt {attempts + 1}")
            response = requests.get(get_url, headers=self.headers)
            if response.status_code != 200:
                print(f"Failed to fetch session status: {response.status_code}, {response.text}")
                return ""
            response_json = response.json()
            if response_json["status_enum"] in ["blocked", "stopped"]:
                return response_json["structured_output"].get("extracted_code", "")
            time.sleep(min(backoff, 60))
            backoff = min(backoff * 3, 60)
            attempts += 1

    def execute(self, state: State) -> dict:
        # devin_prompt = getattr(state, self.input_key[0])
        devin_prompt = state["devin_prompts"]
        # github_url = getattr(state, self.input_key[1])
        github_url = state["github_url"]
        # add_method_text = getattr(state, self.input_key[2])
        add_method_text = state["add_method_text"]
        session_id = self._create_session(devin_prompt, github_url, add_method_text)
        time.sleep(120)
        extracted_code = self._get_devin_response(session_id)
        return {
            self.output_key[0]: extracted_code
        }


if __name__ == "__main__":
    retriever = RetrieveCodeWithDevinNode(
        input_key=["devin_prompts", "github_url", "add_method_text"],
        output_key=["extracted_code"],
    )
    state = {
        "devin_prompts": """
Extract the code related to the contents of the “Description of Methodology” given below from the repository at the “GitHub Repository URL”.
The extracted code and description should be output as “extracted_code”.
If there is no code, output “No applicable code”.""",
        "github_url": "https://github.com/AtheMathmo/AggMo",
        "add_method_text": """
## Method Explanation: Aggregated Momentum
**Introduction to the Problem:**
Momentum methods play a crucial role in gradient-based optimization by aiding optimizers to gain speed in low curvature directions without causing instability in directions with high curvature. The performance of these methods pivots on the choice of damping coefficient (β), which balances speed and stability. High β values expedite momentum, yet they risk introducing oscillations and instabilities. The learning rate often needs to be reduced, decelerating convergence overall to mitigate these challenges.
**Proposed Solution - Aggregated Momentum (AggMo):**
To tackle these oscillations without sacrificing high terminal velocity, the authors introduce Aggregated Momentum (AggMo). This method innovatively maintains several momentum velocity vectors, each with distinct β values. During the update process, these velocities are averaged to perform the final update on parameters.
- **Mechanism:** 
  - Each velocity vector is influenced by a different β parameter, where higher β facilitates speed-up while lower β effectively dampens oscillations.
  - These velocities are then aggregated to determine the update to parameter θ, stabilizing the optimizer while retaining the benefits of large β values.
- **Analogy to Passive Damping:**
  - Similar to a physical structure's use of varied resonant frequency materials to prevent catastrophic resonance, combining multiple velocities in AggMo achieves a system where oscillations are minimized. 
  - This passive damping metaphor elucidates AggMo's stability, mirroring smart material design in engineering, to curtail perilous oscillatory feedback amidst optimization.
- **Implementation**:
  - AggMo is straightforward to incorporate with nearly no computational overhead.
  - By utilizing several damping coefficients, the method improves optimization over ill-conditioned curvature.
- **Reinterpretation of Nesterov's Accelerated Gradient:** 
  - AggMo reinterprets Nesterov's accelerated gradient descent as a specific instance within its framework, showcasing greater theoretical flexibility and more generalized applicability.
**Theoretical Convergence Analysis:**
- AggMo is theoretically analyzed for its rapid convergence on quadratic objectives.
- It successfully achieves a bounded regret in online convex programming, ensuring robust convergence behavior across different convex cost functions.
**Empirical Validation:**
- AggMo serves as an effective drop-in replacement for other momentum techniques, frequently providing faster convergence without necessitating intensive hyperparameter tuning.
- A broad range of tasks including deep autoencoders, convolutional networks, and LSTMs demonstrate AggMo's efficiency and superiority in convergence speed and stability.
**Conclusion:**
Aggregated Momentum elegantly balances speed and stability in gradient-based optimization by averaging velocity vectors governed by diverse damping coefficients. This method curtails oscillatory behavior, facilitating rapid convergence and demonstrating robustness across multiple machine learning frameworks.
"""
    }
    output = retriever.execute(state)
    print(output["extracted_code"])
