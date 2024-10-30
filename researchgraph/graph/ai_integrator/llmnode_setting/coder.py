coder1_setting = {
    "input": [
        "objective",
        "llm_script",
        "method_1_code",
        "method_1_text",
    ],
    "output": ["method_1_experimental_code"],
    "prompt": """<rule>
You are a researcher working on machine learning.
- Tag Descriptions
    - The <objective> tag indicates the objective of the research being undertaken.
    - The <llm_script> tag contains the script used to conduct the experiment.
    - The <method_1_text> tag provides a description of the method used.
    - The <method_1_code> tag contains the code relevant to the method.
- Please follow the rules below for the output:
    - Please rewrite the script enclosed in the <llm_script> tag using the code enclosed in the <method_1_code> tag.
    - Please use the information enclosed in the <method_1_text> tag as a reference when rewriting the script.
    - Please enclose the output in the <method_1_experimental_code> tag when outputting.
    - Please use only Python code for <method_1_experimental_code>.
</rule>
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

coder2_setting = {
    "input": [
        "objective",
        "llm_script",
        "new_method_code",
        "new_method_text",
    ],
    "output": ["new_method_experimental_code"],
    "prompt": """<rule>
You are a researcher working on machine learning.
- Tag Descriptions
    - The <objective> tag indicates the objective of the research being undertaken.
    - The <llm_script> tag contains the script used to conduct the experiment.
    - The <new_method_text> tag provides a description of the method used.
    - The <new_method_code> tag contains the code relevant to the method.
- Please follow the rules below for the output:
    - Please rewrite the script enclosed in the <llm_script> tag using the code enclosed in the <new_method_code> tag.
    - Please use the information enclosed in the <new_method_text> tag as a reference when rewriting the script.
    - Please enclose the output in the <new_method_experimental_code> tag when outputting.
    - Please use only Python code for <new_method_experimental_code>.
</rule>
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
