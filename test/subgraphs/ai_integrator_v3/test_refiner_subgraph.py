from researchgraph.graphs.ai_integrator.ai_integrator_v3.refiner_subgraph.main import RefinerSubgraph
from researchgraph.graphs.ai_integrator.ai_integrator_v3.refiner_subgraph.input_data import refiner_subgraph_input_data
from researchgraph.graphs.ai_integrator.ai_integrator_v3.refiner_subgraph.llmnode_prompt import (
    ai_integrator_v3_llmcreator_prompt,
    ai_integrator_v3_llmcoder_prompt,

)

def test_refiner_subgraph():
    llm_name = "gpt-4o-2024-08-06"
    refiner_subgraph = RefinerSubgraph(
        llm_name=llm_name,
        ai_integrator_v3_llmcreator_prompt=ai_integrator_v3_llmcreator_prompt,
        ai_integrator_v3_llmcoder_prompt=ai_integrator_v3_llmcoder_prompt,
    )
    
    assert refiner_subgraph(state = refiner_subgraph_input_data, )
