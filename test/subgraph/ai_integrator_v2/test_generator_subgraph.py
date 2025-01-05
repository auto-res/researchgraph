from researchgraph.graphs.ai_integrator.ai_integrator_v2.generator_subgraph.main import GeneratorSubgraph
from researchgraph.graphs.ai_integrator.ai_integrator_v2.generator_subgraph.input_data import generator_subgraph_input_data
from researchgraph.graphs.ai_integrator.ai_integrator_v2.generator_subgraph.llmnode_prompt import (
    ai_integrator_v2_extractor_prompt,
    ai_integrator_v2_codeextractor_prompt,
    ai_integrator_v2_creator_prompt,
)


def test_generator_subgraph():
    llm_name = "gpt-4o-2024-08-06"
    save_dir = "/workspaces/researchgraph/data"
    generate_subgraph = GeneratorSubgraph(
        llm_name=llm_name,
        save_dir=save_dir,
        ai_integrator_v1_extractor_prompt=ai_integrator_v2_extractor_prompt,
        ai_integrator_v1_codeextractor_prompt=ai_integrator_v2_codeextractor_prompt,
        ai_integrator_v1_creator_prompt=ai_integrator_v2_creator_prompt,
    )
    
    assert generate_subgraph(state = generator_subgraph_input_data, )
