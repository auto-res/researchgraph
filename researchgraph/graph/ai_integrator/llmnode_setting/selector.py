selector1_setting = {
    "input": ["objective", "environment", "collection_of_papers_1"],
    "output": ["selected_paper_1"],
    "prompt": """
    You are an expert researcher tasked with selecting the most relevant paper for a specific research objective and environment. 
    Given the following objective and environment, review the provided papers and choose the one that best aligns with the requirements. 
    Please enclose the selected paper in the selected_paper_1 tag when outputting.
    <objective>
    {objective}
    </objective>
    <environment>
    {environment}
    </environment>
    <collection_of_papers_1>
    {collection_of_papers_1}
    </collection_of_papers_1>
    <EOS></EOS>
    """,
}

selector2_setting = {
    "input": ["objective", "environment", "collection_of_papers_2"],
    "output": ["selected_paper_2"],
    "prompt": """
    You are an expert researcher tasked with selecting the most relevant paper for a specific research objective and environment. 
    Given the following objective and environment, review the provided papers and choose the one that best aligns with the requirements. 
    Please enclose the selected paper in the selected_paper_2 tag when outputting.
    <objective>
    {objective}
    </objective>
    <environment>
    {environment}
    </environment>
    <collection_of_papers_2>
    {collection_of_papers_2}
    </collection_of_papers_2>
    <EOS></EOS>
    """,
}
