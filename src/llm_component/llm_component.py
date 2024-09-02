# %%
import json
from llmlinks.function import LLMFunction
from llmlinks.llm_client import LLMClient


class LLMComponent:
    def __init__(self, llm_name, json_file_path=None, json_data=None):
        self.llm_name = llm_name
        if json_file_path:
            with open(json_file_path, 'r') as file:
                self.json_data = json.load(file)
        elif json_data:
            if isinstance(json_data, str):
                self.json_data = json.loads(json_data)
            elif isinstance(json_data, dict):
                self.json_data = json_data
        else:
            raise ValueError("Either json_file_path or json_data must be provided.")
        self.input = self.json_data.get('input')
        self.output = self.json_data.get('output')
        self.prompt_template = self.json_data.get('prompt')
        print("LLMComponent initialized")
        print(f"input: {self.input}")
        print(f"output: {self.output}")

    def __call__(self, memory_):
        """LLMComponentの実行

        Args:
            memory_ (_type_): _description_

        Returns:
            _type_: _description_
        """
        llm = LLMClient(self.llm_name)
        
        # LLMを複数回実行する場合
        if isinstance(self.input[0], list):
            num_loop = len(self.input)
            for i in range(num_loop):
                prompt = self.prompt_template[i]
                func = LLMFunction(
                    llm, 
                    prompt,
                    self.input[i],
                    self.output[i],
                )
                
                kwargs = {key: memory_[key] for key in self.input[i]}
                response = func(**kwargs)
                print(response)
                for key in self.output[i]:
                    memory_[key] = response[key][0]
            
        # LLMを一回だけ実行する場合
        else:
            func = LLMFunction(
                llm, 
                self.prompt_template,
                self.input,
                self.output,
            )
            
            kwargs = {key: memory_[key] for key in self.input}
            response = func(**kwargs)
            for key in self.output:
                memory_[key] = response[key][0]
        return memory_


if __name__ == "__main__":
    memory = {
        'source': 'Hello World!!',
        'language': 'japanese',
        'output': None
        }
        
    # 基本となる処理の実行
    llm_name = 'gpt-4o-2024-08-06'
    json_file_path = './test/base.json'
    translate = LLMComponent(llm_name=llm_name, json_file_path=json_file_path)
    memory1 = translate(memory.copy())
    print(memory1)
    
    # outputが二つある場合の実行
    llm_name = 'gpt-4o-2024-08-06'
    json_file_path = './test/two_outputs.json'
    translate2 = LLMComponent(llm_name=llm_name, json_file_path=json_file_path)
    memory2 = translate2(memory.copy())
    print(memory2)
    
    # llmの実行が2回ある場合の実行
    llm_name = 'gpt-4o-2024-08-06'
    json_file_path = './test/two_runs.json'
    translate3 = LLMComponent(llm_name=llm_name, json_file_path=json_file_path)
    memory3 = translate3(memory.copy())
    print(memory3)

    # jsonデータを直接渡す場合
    llm_name = 'gpt-4o-2024-08-06'
    json_data = {
        "input_variables": ["source", "language"],
        "output_variables": ["output"],
        "prompt_template": "<source_text>\n{source}\n</source_text>\n<target_language>\n{language}\n</target_language>\n<rule>\nsource_text タグで与えられた文章を target_language で指定された言語に翻訳して output タグを用いて出力せよ．\n</rule>"
    }
    translate4 = LLMComponent(llm_name=llm_name, json_data=json_data)
    memory4 = translate4(memory.copy())
    print(memory4)
