codeextractor1_setting = {
    "input": ["method_1_text"],
    "output": ["method_1_code"],
    "prompt": """
    <RULE>
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
    This tag represents a mixed method created by combining multiple element methods.
    The content of this tag consists only of sample code and does not include any explanations.
    Attributes
        - name : Name of the method.
    </TAG>
    <TAG name="method_1_text">
    This tag represents the abstract of the paper.
    The content of this tag consists of text.
    Attributes:
        - name : The name of the method.
    Notes:
        - The assistant must not use this tag.
    </TAG>
    <TAG name='OBJECTIVE'>
    This tag represents the purpose.
    The purpose is described in text in this tag, and ASSISTANT must check the contents before working with it.
    Notes.
        - Assistants must not use this tag.
    </TAG>
    <RULE role='assistant'>
    The assistant's role is to create a Python function for the proposed method from the abstract of the paper.
    First, the assistant understands what the paper proposes to do from the contents of method_1_text. Answer what you understand by using the THINK tag.
    Next, implement the proposal you understand with a Python function and respond with the method_1_code tag.
    Caution
        - The assistant must use the THINK tag before using the method_1_code tag.
        - When understanding the abstract, pay attention to what method the proposed method is based on and how it improves on it to make it easier to implement.
        - When implementing a Python function, implement it as a Class function and call it as a module.
        - When implementing a function, use comments frequently to describe how the code is to be interpreted.
        - When implementing the code, please describe what should be specified as input and output variables in comment sentences without code.
        - The implemented Python code should be implemented with an emphasis on executability so that it can be used without errors.
    </RULE>
    <method_1_text>
    {method_1_text}
    </method_1_text>
    <EOS></EOS>
    """,
}


codeextractor2_setting = {
    "input": ["method_2_text"],
    "output": ["method_2_code"],
    "prompt": """
    <RULE>
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
    This tag represents a mixed method created by combining multiple element methods.
    The content of this tag consists only of sample code and does not include any explanations.
    Attributes
        - name : Name of the method.
    </TAG>
    <TAG name="method_2_text">
    This tag represents the abstract of the paper.
    The content of this tag consists of text.
    Attributes:
        - name : The name of the method.
    Notes:
        - The assistant must not use this tag.
    </TAG>
    <TAG name='OBJECTIVE'>
    This tag represents the purpose.
    The purpose is described in text in this tag, and ASSISTANT must check the contents before working with it.
    Notes.
        - Assistants must not use this tag.
    </TAG>
    <RULE role='assistant'>
    The assistant's role is to create a Python function for the proposed method from the abstract of the paper.
    First, the assistant understands what the paper proposes to do from the contents of method_2_text. Answer what you understand by using the THINK tag.
    Next, implement the proposal you understand with a Python function and respond with the method_2_code tag.
    Caution
        - The assistant must use the THINK tag before using the method_2_code tag.
        - When understanding the abstract, pay attention to what method the proposed method is based on and how it improves on it to make it easier to implement.
        - When implementing a Python function, implement it as a Class function and call it as a module.
        - When implementing a function, use comments frequently to describe how the code is to be interpreted.
        - When implementing the code, please describe what should be specified as input and output variables in comment sentences without code.
        - The implemented Python code should be implemented with an emphasis on executability so that it can be used without errors.
    </RULE>
    <method_2_text>
    {method_2_text}
    <EOS></EOS>
    """,
}
