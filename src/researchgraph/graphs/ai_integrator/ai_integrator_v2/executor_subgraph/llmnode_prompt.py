ai_integrator_v2_modifier_prompt = """
You are an engineer who can code machine learning in Python.
Please modify the code of the newly created method given in <new_method_code> below based on the results of the <error_logs>.
Do not make any additional modifications.
Output only the modified code.
<new_method_code>
{{new_method_code}}
</new_method_code>
<error_logs>
{{error_logs}}
</error_logs>
"""
