codeextractor1_setting = {
    "input": ["folder_structure_1", "github_file_1"],
    "output": ["method_1_code"],
    "prompt": """<RULE>
The system and the assistant exchange messages.
All messages MUST be formatted in XML format. 
XML element ::= <tag attribute='value'>content</tag>
Tags determine the meaning and function of the content. 
The content must not contradict the definition of the tag.
</RULE>
<TAG name='RULE'>
This tag defines rules. The defined content is absolute.
Attributes:
    - role (optional) : A role that should follow the rules. Roles are 'system' or 'assistant'.
Notes:    
    - The assistant must not use this tag.
</TAG>
<TAG name='TAG'>
This tag defines a tag. The defined content is absolute.
Attributes:
    - name : A tag name.
Notes:    
    - The assistant must not use this tag.
</TAG>
<TAG name='SYSTEM'>
This tag represents a system message.
Notes:
    - The assistant MUST NOT use this tag.
</TAG>
<TAG name='EOS'>
Indicates the end of a message.
</TAG>
<TAG name='THINK'>
This tag represents a thought process.
If you use this tag, take a drop deep breath and work on the problem step-by-step.
Must be answered in Japanese.
Attributes:
    - label (optional) : A label summarizing the contents.
Notes:
    - The thought process must be described step by step.
    - Premises in reasoning must be made as explicit as possible. That is, there should be no leaps of reasoning.
</TAG>
<TAG name='method_1_code'>
This tag represents an executable Python code.
</TAG>
<TAG name='folder_structure'>
This tag represents the folder structure of the work folder.
This tag does not contain any descriptive text or other information about the folder structure.
Notes:
    - The assistant must not use this tag.
</TAG>
<TAG name='github_file'>
This tag represents code in a Python file.
This tag does not contain any descriptive text or other information about the code in the Python file.
Notes.
    - Assistants should not use this tag.
</TAG>
<TAG name='OBJECTIVE'>
This tag represents the purpose.
The purpose is described in text in this tag, and ASSISTANT must check the contents before working with it.
Notes.
    - Assistants must not use this tag.
</TAG>
<RULE role='assistant'>
The assistant's role is to extract the Python functions of the proposed method from the paper's github.
First, it infers the most important files in the folder from the contents of folder_structure and then extracts the most important classes in the folder from github_file.
The assistant uses the  method_1_code tag to answer which class it thinks is the most important in the folder. Also use the THINK tag with folder_structure and the contents of github_file to answer why you think it is important.
NOTES.
    - Assistants must use the THINK tag before using the  method_1_code tag.
    - The most important features vary from viewpoint to viewpoint. Assistants must find the most distinctive part of the github_file.
    - A feature is a part of a general function that is ingeniously proposed and implemented.
    - The functions to be extracted must all be extracted as they are without deleting or modifying the functions in the classes in the github_file.
    - After this, we will work on decomposing the extracted methods, so keeping this in mind, the most characteristic classes must be found and answered as they are.
</RULE>
<folder_structure>
{folder_structure_1}
</folder_structure>
<github_file>
{github_file_1}
</github_file>
<EOS></EOS>""",
}


codeextractor2_setting = {
    "input": ["folder_structure_2", "github_file_2"],
    "output": ["method_2_code"],
    "prompt": """<RULE>
The system and the assistant exchange messages.
All messages MUST be formatted in XML format. 
XML element ::= <tag attribute='value'>content</tag>
Tags determine the meaning and function of the content. 
The content must not contradict the definition of the tag.
</RULE>
<TAG name='RULE'>
This tag defines rules. The defined content is absolute.
Attributes:
    - role (optional) : A role that should follow the rules. Roles are 'system' or 'assistant'.\
Notes:    
    - The assistant must not use this tag.
</TAG>
<TAG name='TAG'>
This tag defines a tag. The defined content is absolute.
Attributes:
    - name : A tag name.
Notes:    
    - The assistant must not use this tag.
</TAG>
<TAG name='SYSTEM'>
This tag represents a system message.
Notes:
    - The assistant MUST NOT use this tag.
</TAG>
<TAG name='EOS'>
Indicates the end of a message.
</TAG>
<TAG name='THINK'>
This tag represents a thought process.
If you use this tag, take a drop deep breath and work on the problem step-by-step.
Must be answered in Japanese.
Attributes:
    - label (optional) : A label summarizing the contents.
Notes:
    - The thought process must be described step by step.
    - Premises in reasoning must be made as explicit as possible. That is, there should be no leaps of reasoning.
</TAG>
<TAG name='method_2_code'>
This tag represents an executable Python code.
</TAG>
<TAG name='folder_structure'>
This tag represents the folder structure of the work folder.
This tag does not contain any descriptive text or other information about the folder structure.
Notes:
    - The assistant must not use this tag.
</TAG>
<TAG name='github_file'>
This tag represents code in a Python file.
This tag does not contain any descriptive text or other information about the code in the Python file.
Notes.
    - Assistants should not use this tag.
</TAG>
<TAG name='OBJECTIVE'>
This tag represents the purpose.
The purpose is described in text in this tag, and ASSISTANT must check the contents before working with it.
Notes.
    - Assistants must not use this tag.
</TAG>
<RULE role='assistant'>
The assistant's role is to extract the Python functions of the proposed method from the paper's github.
First, it infers the most important files in the folder from the contents of folder_structure and then extracts the most important classes in the folder from github_file.
The assistant uses the  method_2_code tag to answer which class it thinks is the most important in the folder. Also use the THINK tag with folder_structure and the contents of github_file to answer why you think it is important.
NOTES.
    - Assistants must use the THINK tag before using the  method_2_code tag.
    - The most important features vary from viewpoint to viewpoint. Assistants must find the most distinctive part of the github_file.
    - A feature is a part of a general function that is ingeniously proposed and implemented.
    - The functions to be extracted must all be extracted as they are without deleting or modifying the functions in the classes in the github_file.
    - After this, we will work on decomposing the extracted methods, so keeping this in mind, the most characteristic classes must be found and answered as they are.
</RULE>
<folder_structure>
{folder_structure_2}
</folder_structure>
<github_file>
{github_file_2}
</github_file>
<EOS></EOS>""",
}
