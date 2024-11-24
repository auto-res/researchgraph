creator_setting = {
    "input": [
        "objective",
        "add_method_text",
        "add_method_code",
        "base_method_text",
        "base_method_code",
    ],
    "output": ["new_method_text", "new_method_code"],
    "prompt": """<RULE>
You are a researcher working on machine learning.
Please check the descriptions of the tags listed in Tag Descriptions and follow the instructions.
- Tag Descriptions
    - The text enclosed within the <objective> tag indicates the objective of the research being undertaken.
    - The text enclosed within the <add_method_text> tag contains an explanation of a method extracted from a machine learning paper.
    - The text enclosed within the <add_method_code> tag contains the code extracted from the paper.
    - The text enclosed within the <base_method_text> tag provides a description of the base method.
    - The text enclosed within the <base_method_code> tag contains the code of the base method.
- Please follow the rules below to output the code and description of the new method.
    - Please apply the code enclosed in the <add_method_code> tag to the code enclosed in the <base_method_code> tag to generate a new method.
    - Please generate a method that is considered to be novel.
    - Please make sure that the new method protects the content enclosed in the <objective> tag.
    - When generating a new method, please also consider the description of the <add_method_code> enclosed in the <add_method_text> tag and the description of the <base_method_code> enclosed in the <base_method_text> tag.
    - Please enclose the new method in the <new_method_text> tag.
    - Please enclose the code of the new method in the <new_method_code> tag.
    - The output of new_method_code must follow the template enclosed in the <method_template> tag.
</RULE>
<objective>
{objective}
</objective>
<add_method_text>
{add_method_text}
</add_method_text>
<add_method_code>
{add_method_code}
</add_method_code>
<base_method_text>
{base_method_text}
</base_method_text>
<base_method_code>
{base_method_code}
</base_method_code>
<method_template>
{method_template}
</method_template>
<EOS></EOS>""",
}
