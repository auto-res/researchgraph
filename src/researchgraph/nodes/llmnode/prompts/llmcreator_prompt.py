#TODO: 本プロンプトは以下のパスに移行予定です. 
# researchgraph/graphs/ai_integrator/ai_integrator_v1/llm_node_prompt.py

llmcreator_prompt = """
You are an expert in machine learning. Based on the following method description and code, generate {{ num_ideas }} innovative ideas to improve the method.

### Method Description:
{{ base_method_text }}

### Method Code:
{{ base_method_code }}

### Output Format:
[
    {
        "idea": "<Brief Idea>",
        "description": "<Detailed Explanation>",
        "score": <Numeric Score>
    },
    ...
]

"""