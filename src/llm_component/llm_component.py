# %%
import re
import json
from llmlinks.function import LLMFunction



def llm_component(memory, llm_name, json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    input_variables = data.get('input_variables')
    output_variables = data.get('output_variables')
    prompt_template = data.get('prompt_template')
    print(f"input_variables: {input_variables}")
    print(f"output_variables: {output_variables}")
    func = LLMFunction(
        llm_name, 
        prompt_template,
        input_variables,
        output_variables,
        )
    
    kwargs = {key: memory[key] for key in input_variables}
    print(f"kwargs: {kwargs}")
    response = func(**kwargs)

    # outputが一種類の場合    
    for key in output_variables:
        output = re.search(fr'<{key}>(.*?)</{key}>', response[f'{key}'][0]).group(1)
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
