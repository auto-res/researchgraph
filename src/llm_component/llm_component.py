# %%
import json
from llmlinks.function import LLMFunction



def llm_component(memory, llm_name, json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    input_variables = data.get('input_variables')
    output_variables = data.get('output_variables')
    prompt_template = data.get('prompt_template')
    func = LLMFunction(
        llm_name, 
        prompt_template,
        input_variables,
        output_variables,
        )
    
    kwargs = {key: memory[key] for key in input_variables}
    print(f"kwargs: {kwargs}")
    output = func(**kwargs)
    print(f"output: {output}")
    
    for key in output_variables:
        memory[key] = output
    print(f"memory: {memory}")
    return memory


if __name__ == "__main__":
   memory = {
       'source': 'Hello World!!',
       'language': 'japanese',
       'output': None
       }
   llm_name = 'gemini-1.5-pro'
   json_file_path = './test/template.json'
   memory = llm_component(memory, llm_name, json_file_path)
