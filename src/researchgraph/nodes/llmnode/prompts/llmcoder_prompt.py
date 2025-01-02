#TODO: 本プロンプトは以下のパスに移行予定です. 
# researchgraph/graphs/ai_integrator/ai_integrator_v1/llm_node_prompt.py

llmcoder_prompt = """
You are an expert programmer and researcher. Given the following method and an improvement idea, implement the improved version of the method.

### Method Description:
{{ base_method_text }}

### Method Code:
{{ base_method_code }}

### Improvement Idea:
{{ selected_idea }}

### Format:
{
    "improved_base_method_text": "Improved Method Description",
    "improved_base_method_code": "Improved Method Code"
}

"""