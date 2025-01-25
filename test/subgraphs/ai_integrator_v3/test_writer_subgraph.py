import os
from researchgraph.graphs.ai_integrator.ai_integrator_v3.writer_subgraph.main import WriterSubgraph
from researchgraph.graphs.ai_integrator.ai_integrator_v3.writer_subgraph.input_data import writer_subgraph_input_data

GITHUB_WORKSPACE = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
TEST_TEMPLATE_DIR = os.path.join(GITHUB_WORKSPACE, "src/researchgraph/graphs/ai_scientist/templates/2d_diffusion")
TEST_FIGURES_DIR = os.path.join(GITHUB_WORKSPACE, "images")


def test_writerr_subgraph():
    llm_name = "gpt-4o-2024-08-06"
    writer_subgraph = WriterSubgraph(
        llm_name=llm_name,
        template_dir = TEST_TEMPLATE_DIR, 
        figures_dir = TEST_FIGURES_DIR, 
    )
    
    assert writer_subgraph(state = writer_subgraph_input_data, )
