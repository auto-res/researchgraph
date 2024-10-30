codeextractor1_setting = {
    "input": ["method_1_text", "folder_structure_1", "github_file_1"],
    "output": ["method_1_code"],
    "prompt": """<RULE>
You are a researcher working on machine learning.
- Tag Descriptions
    - The text enclosed within the <method_1_text> tag contains an explanation of a method extracted from a machine learning paper.
    - The text enclosed within the <folder_structure_1> tag shows the folder structure of the corresponding GitHub repository for the paper.
    - The text enclosed within the <github_file_1> tag contains the code from Python files in the corresponding GitHub repository.
- Instructions for Extracting Python Code
    - Extract the relevant sections of Python code from the content enclosed within the <github_file_1> tag based on the method described in the <method_1_text> tag.
    - Use the folder structure provided within the <folder_structure_1> tag as a reference when extracting the code.
    - Please extract any code that seems to be related.
    - Enclose the extracted code within <method_1_code> tags.
    - If no corresponding code exists, output "No corresponding code exists." In this case, enclose the output within <method_1_code> tags.
</RULE>
<method_1_text>
{method_1_text}
</method_1_text>
<folder_structure>
{folder_structure_1}
</folder_structure>
<github_file>
{github_file_1}
</github_file>
<EOS></EOS>""",
}


codeextractor2_setting = {
    "input": ["method_2_text", "folder_structure_2", "github_file_2"],
    "output": ["method_2_code"],
    "prompt": """<RULE>
You are a researcher working on machine learning.
- Tag Descriptions
    - The text enclosed within the <method_2_text> tag contains an explanation of a method extracted from a machine learning paper.
    - The text enclosed within the <folder_structure_2> tag shows the folder structure of the corresponding GitHub repository for the paper.
    - The text enclosed within the <github_file_2> tag contains the code from Python files in the corresponding GitHub repository.
- Instructions for Extracting Python Code
    - Extract the relevant sections of Python code from the content enclosed within the <github_file_2> tag based on the method described in the <method_2_text> tag.
    - Use the folder structure provided within the <folder_structure_2> tag as a reference when extracting the code.
    - Please extract any code that seems to be related.
    - Enclose the extracted code within <method_2_code> tags.
    - If no corresponding code exists, output "No corresponding code exists." In this case, enclose the output within <method_2_code> tags.
</RULE>
<method_2_text>
{method_2_text}
</method_2_text>
<folder_structure>
{folder_structure_2}
</folder_structure>
<github_file>
{github_file_2}
</github_file>
<EOS></EOS>""",
}
