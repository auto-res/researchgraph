ai_integrator_v3_llmcreator_prompt = """
<RULE>
You are an expert in machine learning.
- Instructions for Generating Ideas
    - Based on the provided method description and code, generate {{ num_ideas }} innovative ideas to improve the method.
    - Each idea should include a brief summary and a detailed explanation.
    - Ensure that the ideas are feasible and align with modern machine learning practices.
- Output Format
    - Only output the following JSON structure. Do not include any additional text, explanation, or comments.
    - Example Format:
        [
            {
                "idea": "<Brief Idea>",
                "description": "<Detailed Explanation>"
            }, 
            ...
        ]
</RULE>
<base_method_text>
{{ base_method_text }}
</base_method_text>
<base_method_code>
{{ base_method_code }}
</base_method_code>
<EOS></EOS>"""

ai_integrator_v3_llmcoder_prompt = """
<RULE>
You are an expert programmer and researcher.
- Instructions for Refining the Method
    - Using the provided method description, code, and generated ideas, implement a refined version of the method.
    - The refined method should address the generated idea and maintain the original functionality.
- Output Format
    - The output should be a dictionary with two keys:
        - `"refined_method_text"`: A description of the refined method.
        - `"refined_method_code"`: The Python code implementing the refined method.
    - Example Format:
        {
            "refined_method_text": "Refined Method Description",
            "refined_method_code": "Refined Method Code"
        }
</RULE>
<base_method_text>
{{ base_method_text }}
</base_method_text>
<base_method_code>
{{ base_method_code }}
</base_method_code>
<generated_ideas>
{{ generated_ideas }}
</generated_ideas>
<EOS></EOS>"""