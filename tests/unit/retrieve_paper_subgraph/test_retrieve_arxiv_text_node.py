import os
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph

from airas.nodes.retrievenode import RetrievearXivTextNode
# from researchgraph.nodes.retrievenode import RetrieveGithubRepositoryNode


class State(BaseModel):
    arxiv_url: str = Field(default="")
    paper_text: str = Field(default="")
    github_url: str = Field(default="")
    # folder_structure: str = Field(default="")
    # github_file: str = Field(default="")
    add_method_text: str = Field(default="")


SAVE_DIR = os.environ.get("SAVE_DIR", "/workspaces/researchgraph/data")


# NOTE：It is executed by Github actions.
def test_retrieve_arxiv_text_node():
    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "arxivretriever",
        RetrievearXivTextNode(
            input_key=["arxiv_url"],
            output_key=["paper_text"],
            save_dir=SAVE_DIR,
        ),
    )
    graph_builder.add_edge(START, "arxivretriever")
    graph_builder.add_edge("arxivretriever", END)
    graph = graph_builder.compile()

    state = {
        "arxiv_url": "https://arxiv.org/abs/1604.03540v1",
    }
    assert graph.invoke(state, debug=True)


# def test_retrieve_github_repository_node():
#     input_key = ["github_url"]
#     output_key = ["folder_structure", "github_file"]
#     graph_builder = StateGraph(State)
#     graph_builder.add_node(
#         "githubretriever",
#         RetrieveGithubRepositoryNode(
#             input_key=input_key,
#             output_key=output_key,
#             save_dir=SAVE_DIR,
#         ),
#     )
#     graph_builder.set_entry_point("githubretriever")
#     graph_builder.set_finish_point("githubretriever")
#     graph = graph_builder.compile()

#     state = {
#         "github_url": "https://github.com/adelnabli/acid?tab=readme-ov-file/info/refs"
#     }
#     assert graph.invoke(state, debug=True)


# NOTE:Commented out to avoid automatic test execution because of the cost of running Devin during Test in Github actions.
# def test_retrieve_code_with_devin_node():
#     graph_builder = StateGraph(State)
#     graph_builder.add_node(
#         "githubretriever",
#         RetrieveCodeWithDevinNode(
#             input_key = ["github_url", "add_method_text"],
#             output_key=["extracted_code"],
#             )
#     )
#     graph_builder.add_edge(START, "githubretriever")
#     graph_builder.add_edge("githubretriever", END)
#     graph = graph_builder.compile()

#     state = {
#         "github_url": "https://github.com/AtheMathmo/AggMo",
#         "add_method_text": """
# ## Method Explanation: Aggregated Momentum
# **Introduction to the Problem:**
# Momentum methods play a crucial role in gradient-based optimization by aiding optimizers to gain speed in low curvature directions without causing instability in directions with high curvature. The performance of these methods pivots on the choice of damping coefficient (β), which balances speed and stability. High β values expedite momentum, yet they risk introducing oscillations and instabilities. The learning rate often needs to be reduced, decelerating convergence overall to mitigate these challenges.
# **Proposed Solution - Aggregated Momentum (AggMo):**
# To tackle these oscillations without sacrificing high terminal velocity, the authors introduce Aggregated Momentum (AggMo). This method innovatively maintains several momentum velocity vectors, each with distinct β values. During the update process, these velocities are averaged to perform the final update on parameters.
# - **Mechanism:**
#   - Each velocity vector is influenced by a different β parameter, where higher β facilitates speed-up while lower β effectively dampens oscillations.
#   - These velocities are then aggregated to determine the update to parameter θ, stabilizing the optimizer while retaining the benefits of large β values.
# - **Analogy to Passive Damping:**
#   - Similar to a physical structure's use of varied resonant frequency materials to prevent catastrophic resonance, combining multiple velocities in AggMo achieves a system where oscillations are minimized.
#   - This passive damping metaphor elucidates AggMo's stability, mirroring smart material design in engineering, to curtail perilous oscillatory feedback amidst optimization.
# - **Implementation**:
#   - AggMo is straightforward to incorporate with nearly no computational overhead.
#   - By utilizing several damping coefficients, the method improves optimization over ill-conditioned curvature.
# - **Reinterpretation of Nesterov's Accelerated Gradient:**
#   - AggMo reinterprets Nesterov's accelerated gradient descent as a specific instance within its framework, showcasing greater theoretical flexibility and more generalized applicability.
# **Theoretical Convergence Analysis:**
# - AggMo is theoretically analyzed for its rapid convergence on quadratic objectives.
# - It successfully achieves a bounded regret in online convex programming, ensuring robust convergence behavior across different convex cost functions.
# **Empirical Validation:**
# - AggMo serves as an effective drop-in replacement for other momentum techniques, frequently providing faster convergence without necessitating intensive hyperparameter tuning.
# - A broad range of tasks including deep autoencoders, convolutional networks, and LSTMs demonstrate AggMo's efficiency and superiority in convergence speed and stability.
# **Conclusion:**
# Aggregated Momentum elegantly balances speed and stability in gradient-based optimization by averaging velocity vectors governed by diverse damping coefficients. This method curtails oscillatory behavior, facilitating rapid convergence and demonstrating robustness across multiple machine learning frameworks.
# """
#     }

#     assert graph.invoke(state, debug=True)
