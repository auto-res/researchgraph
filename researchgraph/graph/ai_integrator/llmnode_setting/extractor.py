extractor_setting = {
    "input": ["paper_text"],
    "output": ["add_method_text"],
    "prompt": """<rule>
You are a researcher working on machine learning.
The following <paper_text> tags enclose the full text data of the paper.
Please extract the specific method(s) claimed in the paper and output them between the <add_method_text> tags.
</rule>
<paper_text>
{paper_text}
</paper_text>""",
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
