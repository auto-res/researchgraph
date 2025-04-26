import os
from airas.graphs.ai_integrator.ai_integrator_v3.base_paper_subgraph.main import (
    BasePaperSubgraph,
)
from airas.graphs.ai_integrator.ai_integrator_v3.base_paper_subgraph.input_data import (
    base_paper_subgraph_input_data,
)
from airas.graphs.ai_integrator.ai_integrator_v3.base_paper_subgraph.llmnode_prompt import (
    ai_integrator_v3_select_paper_prompt,
    ai_integrator_v3_summarize_paper_prompt,
)

GITHUB_WORKSPACE = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
SAVE_DIR = os.path.join(GITHUB_WORKSPACE, "data")


def test_base_paper_subgraph():
    llm_name = "gpt-4o-2024-08-06"
    num_retrieve_paper = 3
    period_days = 90
    save_dir = SAVE_DIR
    api_type = "arxiv"
    base_paper_subgraph = BasePaperSubgraph(
        llm_name=llm_name,
        num_retrieve_paper=num_retrieve_paper,
        period_days=period_days,
        save_dir=save_dir,
        api_type=api_type,
        ai_integrator_v3_select_paper_prompt=ai_integrator_v3_select_paper_prompt,
        ai_integrator_v3_summarize_paper_prompt=ai_integrator_v3_summarize_paper_prompt,
    )

    assert base_paper_subgraph(
        state=base_paper_subgraph_input_data,
    )
