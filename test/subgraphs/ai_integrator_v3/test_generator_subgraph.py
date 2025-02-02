import os
from researchgraph.graphs.ai_integrator.ai_integrator_v3.generator_subgraph.generator_subgraph import (
    GeneratorSubgraph,
)
from researchgraph.graphs.ai_integrator.ai_integrator_v3.generator_subgraph.input_data import (
    generator_subgraph_input_data,
)
from researchgraph.graphs.ai_integrator.ai_integrator_v3.generator_subgraph.llmnode_prompt import (
    ai_integrator_v3_extractor_prompt,
    ai_integrator_v3_codeextractor_prompt,
    ai_integrator_v3_creator_prompt,
)

GITHUB_WORKSPACE = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
SAVE_DIR = os.path.join(GITHUB_WORKSPACE, "data")


def test_generator_subgraph():
    llm_name = "gpt-4o-2024-08-06"
    save_dir = SAVE_DIR
    generate_subgraph = GeneratorSubgraph(
        llm_name=llm_name,
        save_dir=save_dir,
        ai_integrator_v3_extractor_prompt=ai_integrator_v3_extractor_prompt,
        ai_integrator_v3_codeextractor_prompt=ai_integrator_v3_codeextractor_prompt,
        ai_integrator_v3_creator_prompt=ai_integrator_v3_creator_prompt,
    )

    assert generate_subgraph(
        state=generator_subgraph_input_data,
    )
