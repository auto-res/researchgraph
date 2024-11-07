comparator_setting = {
    "input": [
        "method_1_score",
        "method_1_completion",
        "method_2_score",
        "method_2_completion",
    ],
    "output": [
        "comparison_result",
        "comparison_result_content",
    ],
    "prompt": """
    <rule>
    
    </rule>
    <method_1_score>
    {method_1_score}
    </method_1_score>
    <method_1_completion>
    {method_1_completion}
    </method_1_completion>
    <method_2_score>
    {method_2_score}
    </method_2_score>
    <method_2_completion>
    {method_2_completion}
    </method_2_completion>
    """,
}
