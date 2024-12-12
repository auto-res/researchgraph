ai_integrator_v1_extractor_prompt = """
You are a researcher working on machine learning.
The following <paper_text> tags enclose the full text data of the paper.
Please extract the explanation of the method introduced in the given paper.
<paper_text>
{{paper_text}}
</paper_text>
"""

ai_integrator_v1_codeextractor_prompt = """
<RULE>
You are a researcher working on machine learning.
- Tag Descriptions
    - The text enclosed within the <add_method_text> tag contains an explanation of a method extracted from a machine learning paper.
    - The text enclosed within the <folder_structure> tag shows the folder structure of the corresponding GitHub repository for the paper.
    - The text enclosed within the <github_file> tag contains the code from Python files in the corresponding GitHub repository.
- Instructions for Extracting Python Code
    - Extract the relevant sections of Python code from the content enclosed within the <github_file> tag based on the method described in the <add_method_text> tag.
    - Use the folder structure provided within the <folder_structure> tag as a reference when extracting the code.
    - Please extract any code that seems to be related.
    - If no corresponding code exists, output "No corresponding code exists."
</RULE>
<add_method_text>
{{add_method_text}}
</add_method_text>
<folder_structure>
{{folder_structure}}
</folder_structure>
<github_file>
{{github_file}}
</github_file>
<EOS></EOS>"""


ai_integrator_v1_creator_prompt = """
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
    - When creating a new method, please also consider the description of the method enclosed in the <add_method_text> tag and the description enclosed in the <base_method_text> tag.
    - Please output the new method you have created as new_method_text.
    - Please output the new code you have created as new_method_code.
    - The output of new_method_code must follow the template enclosed in the <method_template> tag.
</RULE>
<objective>
{{objective}}
</objective>
<add_method_text>
{{add_method_text}}
</add_method_text>
<add_method_code>
{{add_method_code}}
</add_method_code>
<base_method_text>
{{base_method_text}}
</base_method_text>
<base_method_code>
{{base_method_code}}
</base_method_code>
<method_template>
{{method_template}}
</method_template>
<EOS></EOS>"""
