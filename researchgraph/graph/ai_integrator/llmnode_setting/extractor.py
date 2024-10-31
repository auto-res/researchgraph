extractor1_setting = {
    "input": ["paper_text_1"],
    "output": ["method_1_text"],
    "prompt": """<rule>
You are a researcher working on machine learning.
The following <paper_text_1> tags enclose the full text data of the paper.
Please extract the specific method(s) claimed in the paper and output them between the <method_1_text> tags.
</rule>
<paper_text_1>
{paper_text_1}
</paper_text_1>""",
}
extractor2_setting = {
    "input": ["paper_text_2"],
    "output": ["method_2_text"],
    "prompt": """<rule>
You are a researcher working on machine learning.
The following <paper_text_2> tags enclose the full text data of the paper.
Please extract the specific method(s) claimed in the paper and output them between the <method_2_text> tags.
</rule>
<paper_text_2>
{paper_text_2}
</paper_text_2>
<EOS></EOS>""",
}
