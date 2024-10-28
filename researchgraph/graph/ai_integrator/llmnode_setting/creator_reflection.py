creator_reflection_setting = {
    "input": [
        "objective",
        "new_method_text",
        "new_method_code",
    ],
    "output": ["new_method_text", "new_method_code", "reflection_result"],
    "prompt": """
    <RULE>
    The system and the assistant exchange messages.
    All messages MUST be formatted in XML format. 
    XML element ::= <tag attribute='value'>content</tag>
    Tags determine the meaning and function of the content. The content must not contradict the definition of the tag.
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
    <TAG name='new_method_code'>
    This tag is based on the contents of old_method_code, and expresses in code the results examined in new_method_text.
    The contents of this tag are only python code.
    Attributes
        - name : Name of the method.
    </TAG>
    <TAG name='new_method_text'>
    This tag indicates the result of careful consideration and improvement of the quality, novelty, and feasibility of the ideas in the content of old_method_text.
    The content of this tag is text only.
    </TAG>
    <TAG name='old_method_code'>
    This tag represents the code implemented based on the new proposal outline in old_method_text.
    The contents of this tag are only python code.
    Attributes
        - name : Name of the method.
    </TAG>
    <TAG name='old_method_text'>
    This tag represents a new proposal with ideas based on the methods proposed in the paper.
    The content of this tag is text only.
    </TAG>
    <TAG name='objective'>
    This tag represents the purpose.
    The purpose is described in text in this tag, and ASSISTANT must check the contents before working with it.
    Notes.
        - Assistants must not use this tag.
    </TAG>
    <TAG name='reflection_result'>
    This tag describes whether the new proposal should be improved.
    It returns YES if the proposal should be improved, NO if it should not.
    There are no explanatory text or other information other than YES and NO.
    </TAG>
    <RULE role='assistant'>
    The assistant's role is to review the validity of the new proposal considered earlier.
    First, the assistant should carefully read and understand the contents of old_method_text, old_method_code.
    Next, carefully review the old_method_text, old_method_code you have just considered in terms of the quality, novelty, and feasibility of the idea.
    Explain the results of your examination in the THINK tag.
    If, as a result of your examination, points to be improved exist, return YES to the reflection_result tag, and return the proposal and its code with the improvements made using the new_method_text and old_method_code tags.
    If there are no improvements to be made, return NO to the reflection_result tag and return the contents of the old_method_text and old_method_code tags in the new_method_text and new_method_code tags as they are.
    Caution.
        - The evaluation should consider the quality of the idea, novelty, feasibility, and any other factors deemed important in evaluating the idea.
        - Ideas should be clear and concise.
        - It should not overcomplicate things.
        - Unless there are glaring problems, do not make improvements, but respond as is.
    </RULE>
    <objective>
    {objective}
    </objective>
    <old_method_text>
    {new_method_text}
    </old_method_text>
    <old_method_code>
    {new_method_code}
    </old_method_code>
    <EOS></EOS>"
    """,
}
