from pydantic import BaseModel
from litellm import completion
from jinja2 import Environment
import ast


class LLMOutput(BaseModel):
    detailed_description_of_methodology: str
    novelty: str
    experimental_procedure: str
    new_method_code: str


def method_integrate(
    llm_name: str,
    prompt_template: str,
    # objective: str,
    base_method_code: str,
    base_method_text: str,
    add_method_code: str,
    add_method_text: str,
) -> tuple[str, str, str, str]:
    data = {
        # "objective": objective,
        "base_method_code": base_method_code,
        "base_method_text": base_method_text,
        "add_method_code": add_method_code,
        "add_method_text": add_method_text,
    }

    env = Environment()
    template = env.from_string(prompt_template)
    prompt = template.render(data)

    response = completion(
        model=llm_name,
        messages=[
            {"role": "user", "content": f"{prompt}"},
        ],
        response_format=LLMOutput,
    )
    output = response.choices[0].message.content
    output_dict = ast.literal_eval(output)
    new_method_detailed_description_of_methodology = output_dict[
        "detailed_description_of_methodology"
    ]
    new_method_novelty = output_dict["novelty"]
    new_method_experimental_procedure = output_dict["experimental_procedure"]
    new_method_code = output_dict["new_method_code"]
    return (
        new_method_detailed_description_of_methodology,
        new_method_novelty,
        new_method_experimental_procedure,
        new_method_code,
    )


# TODO:手法の合成方法はかなり改善の余地がある．
method_integrate_prompt = """
You are a researcher working on machine learning.
Please check the descriptions of the tags listed in Tag Descriptions and follow the instructions.
- Tag Descriptions
    - The text enclosed within the <add_method_text> tag contains an explanation of a method extracted from a machine learning paper.
    - The text enclosed within the <add_method_code> tag contains the code extracted from the paper.
    - The text enclosed within the <base_method_text> tag provides a description of the base method.
    - The text enclosed within the <base_method_code> tag contains the code of the base method.
- Please follow the rules below to output the code and description of the new method.
    - Think of a possible new method from the methods given in <base_method_text> and <add_method_text>.
    - Please generate a method that is considered to be novel.
    - When creating a new method, please also consider the description of the method enclosed in the <add_method_code> tag and the description enclosed in the <base_method_code> tag.
    - Output a detailed description of the newly conceived method as a detailed_description_of_methodology. Please be as specific as possible.
    - Please output as “novelty” the novelty of the new method you have considered. Please also consider what and how it is novelty.
    - Please come up with an experimental design to validate the new method you have thought of and output it as “experimental_procedure”. The experiments will be conducted based on the experimental_procedure, so please provide as much detail as possible.
    - Please output the new method you have created as new_method_text.
    - Output the code as new_method_code to experiment with the new method you have come up with. The code for the experiment should be output in python. We want as much detailed experimental code as possible so that we can start experimenting immediately.
</RULE>
<add_method_text>
{{add_method_text}}
</add_method_text>
<add_method_code>
{{add_method_code}}
</add_method_code>
<base_method_text>
{{base_method_text}}
</base_method_text>
<base_method_code>
{{base_method_code}}
</base_method_code>
<EOS></EOS>"""
