codeextractor_setting = {
    "input": ["add_method_text", "folder_structure", "github_file"],
    "output": ["add_method_code"],
    "prompt": """<RULE>
You are a researcher working on machine learning.
- Tag Descriptions
    - The text enclosed within the <add_method_text> tag contains an explanation of a method extracted from a machine learning paper.
    - The text enclosed within the <folder_structure> tag shows the folder structure of the corresponding GitHub repository for the paper.
    - The text enclosed within the <github_file> tag contains the code from Python files in the corresponding GitHub repository.
- Instructions for Extracting Python Code
    - Extract the relevant sections of Python code from the content enclosed within the <github_file> tag based on the method described in the <add_method_text> tag.
    - Use the folder structure provided within the <folder_structure> tag as a reference when extracting the code.
    - Please extract any code that seems to be related.
    - Enclose the extracted code within <add_method_code> tags.
    - If no corresponding code exists, output "No corresponding code exists." In this case, enclose the output within <add_method_code> tags.
</RULE>
<add_method_text>
{add_method_text}
</add_method_text>
<folder_structure>
{folder_structure}
</folder_structure>
<github_file>
{github_file}
</github_file>
<EOS></EOS>""",
}
