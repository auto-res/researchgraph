coder1_setting = {
    "input": [
        "objective",
        "environment", 
        "method_1_code",
        "method_1_text",
        ],
    "output": ["method_1_code_experiment"],
    "prompt": 
    """
    <rule>
    </rule>
    <objective>
    {objective}
    </objective>
    <environment>
    {environment}
    </environment>
    <method_1_code>
    {method_1_code}
    </method_1_code>
    <method_1_text>
    {method_1_text}
    </method_1_text>
    
    <template>
    </template>
    """
}

coder2_setting = {
    "input": [
        "objective",
        "environment", 
        "method_2_code",
        "method_2_text",
        ],
    "output": ["method_2_code_experiment"],
    "prompt": 
    """
    <rule>
    </rule>
    <objective>
    {objective}
    </objective>
    <environment>
    {environment}
    </environment>
    <method_2_code>
    {method_2_code}
    </method_2_code>
    <method_2_text>
    {method_2_text}
    </method_2_text>
    
    <template>
    </template>
    """
}
