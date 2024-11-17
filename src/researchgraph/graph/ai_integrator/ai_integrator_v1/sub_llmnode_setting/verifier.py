verifier1_setting = {
    "input": [
        "objective",
        "llm_script",
        "method_1_code",
        "method_1_text",
    ],
    "output": ["method_1_executable"],
    "prompt": """<RULE>
You are a researcher working on machine learning.
- Tag Descriptions
    - The <objective> tag indicates the objective of the research being undertaken.
    - The <llm_script> tag contains the script used to conduct the experiment.
    - The <method_1_text> tag provides a description of the method used.
    - The <method_1_code> tag contains the code relevant to the method.
- Please follow the rules below for the output:
    - Determine whether the script in the <llm_script> tag can be rewritten using the code within the <method_1_code> tag.
    - When making the judgment, refer to the objective of the experiment described in the <objective> tag and the method explanation provided in the <method_1_text> tag.
    - Output either True or False.
    - Enclose the output within the <method_1_executable> tag.
</RULE>
<objective>
{objective}
</objective>
<llm_script>
{llm_script}
</llm_script>
<method_1_text>
{method_1_text}
</method_1_text>
<method_1_code>
{method_1_code}
</method_1_code>
<EOS></EOS>""",
}

verifier2_setting = {
    "input": [
        "objective",
        "llm_script",
        "new_method_code",
        "new_method_text",
    ],
    "output": ["new_method_executable"],
    "prompt": """<RULE>
You are a researcher working on machine learning.
- Tag Descriptions
    - The <objective> tag indicates the objective of the research being undertaken.
    - The <llm_script> tag contains the script used to conduct the experiment.
    - The <new_method_text> tag provides a description of the method used.
    - The <new_method_code> tag contains the code relevant to the method.
- Please follow the rules below for the output:
    - Determine whether the script in the <llm_script> tag can be rewritten using the code within the <new_method_code> tag.
    - When making the judgment, refer to the objective of the experiment described in the <objective> tag and the method explanation provided in the <new_method_text> tag.
    - Output either True or False.
    - Enclose the output within the <new_method_executable> tag.
</RULE>
<objective>
{objective}
</objective>
<llm_script>
{llm_script}
</llm_script>
<new_method_text>
{new_method_text}
</new_method_text>
<new_method_code>
{new_method_code}
</new_method_code>
<EOS></EOS>""",
}
