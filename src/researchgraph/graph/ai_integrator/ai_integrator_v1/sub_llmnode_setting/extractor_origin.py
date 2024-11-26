extractor1_setting = {
    "input": ["selected_paper_1"],
    "output": ["github_url_1", "method_1_text"],
    "prompt": """
    <role>
    You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.
    </role>
    <rule>
    You have been researching about a paper. You have found a paper that are related to the topic. Read the paper abouve and make a report in markdown formatt.
    For github_url_1 you have to answer the repo url for the implementation.
    For method_1_text you habe to summarize the method of the paper. Write as specific as possible as possible including pseudocode or mathematical formula.
    </rule>
    <selected_paper_1>
    {selected_paper_1}
    </selected_paper_1>
    == REPORT EXAMPLE ==
    <github_url_1>
    {github_url_1}
    </github_url_1>
    <method_1_text>
    {method_1_text}
    </method_1_text>
""",
}
extractor2_setting = {
    "input": ["selected_paper_2"],
    "output": ["github_url_2", "method_2_text"],
    "prompt": """
    <role>
    You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.
    </role>
    <rule>
    You have been researching about a paper. You have found a paper that are related to the topic. Read the paper abouve and make a report in markdown formatt.
    For github_url_2 you have to answer the repo url for the implementation.
    For method_2_text you habe to summarize the method of the paper. Write as specific as possible as possible including pseudocode or mathematical formula.
    </rule>
    <selected_paper_2>
    {selected_paper_2}
    </selected_paper_2>
    == REPORT EXAMPLE ==
    <github_url_2>
    {github_url_2}
    </github_url_2>
    <method_2_text>
    {method_2_text}
    </method_2_text>
""",
}
