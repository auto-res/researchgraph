keyworder1_setting = {
    "input": [
        ["environment", "objective"],
        ["environment", "objective", "keywords_mid_thought_1"],
    ],
    "output": [["keywords_mid_thought_1", "keywords_1"], ["keywords_1"]],
    "prompt": [
        """
        You have to think of a 5 KEYWORDs regarding academic search. There is a ojbective and limitation that we can handle, so you have to first interpret what the objective really means in keyword search. Answer step by step what do we need when thinking keywords.== OBJECTIVE ==\n{objective}== LIMITATION ==\n{environment}
        """,
        """
        You have to think of a 5 KEYWORDs in in JSON format. Read all the information and make a report in JSON formatt\n\n You have to write keyword ONLY.\n\n== REPORT EXAMPLE ==\n{report_example}== OBJECTIVE ==\n{objective}== LIMITATION ==\n{environment}== THOUGHT ==\n{keywords_mid_thought}"
        """,
    ],
}


keyworder2_setting = {
    "input": [
        ["environment", "objective"],
        ["environment", "objective", "keywords_mid_thought_2"],
    ],
    "output": [["keywords_mid_thought_2", "keywords_2"], ["keywords_2"]],
    "prompt": [
        """
        You have to think of a 5 KEYWORDs regarding academic search. There is a ojbective and limitation that we can handle, so you have to first interpret what the objective really means in keyword search. Answer step by step what do we need when thinking keywords.== OBJECTIVE ==\n{objective}== LIMITATION ==\n{environment}
        """,
        """
        You have to think of a 5 KEYWORDs in in JSON format. Read all the information and make a report in JSON formatt\n\n You have to write keyword ONLY.\n\n== REPORT EXAMPLE ==\n{report_example}== OBJECTIVE ==\n{objective}== LIMITATION ==\n{environment}== THOUGHT ==\n{keywords_mid_thought}"
        """,
    ],
}
