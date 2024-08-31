from llmlinks.function import LLMFunction


llm_name = 'gemini-1.5-pro'

func = LLMFunction(
    llm_name, 
    prompt_template=prompt_template,
    input_variables=['source', 'language'],
    output_variables=['output']
    )
