from pydantic import BaseModel
from litellm import completion
from jinja2 import Environment
import json

llm_decide_prompt = """
# Instructions:
You determine whether the Python script has succeeded or failed based on the given information. You must output True or False according to the following rules.

# Rules:
- Output False if “Error Data” contains an error. If “Error Date” contains a non-error or empty content, output True.
- Output False if both “Error Data” and “Output Data” are empty.

# Error Data:
{{ error_text_data }}
# Output Data:
{{ output_text_data }}"""


class LLMOutput(BaseModel):
    judgment_result: bool


def llm_decide(
    llm_name: str,
    output_text_data: str,
    error_text_data: str,
    prompt_template: str = llm_decide_prompt,
    max_retries: int = 3,
) -> bool | None:
    data = {"output_text_data": output_text_data, "error_text_data": error_text_data}

    env = Environment()
    template = env.from_string(prompt_template)
    prompt = template.render(data)

    for attempt in range(max_retries):
        try:
            response = completion(
                model=llm_name,
                messages=[
                    {"role": "user", "content": f"{prompt}"},
                ],
                response_format=LLMOutput,
            )
            output = response.choices[0].message.content
            output_dict = json.loads(output)
            judgment_result = output_dict["judgment_result"]
            return judgment_result
        except Exception as e:
            print(f"[Attempt {attempt+1}/{max_retries}] Error calling LLM: {e}")
    print("Exceeded maximum retries for LLM call.")
    return None


if __name__ == "__main__":
    llm_name = "gpt-4o-mini-2024-07-18"
    output_text_data = "No error"
    error_text_data = "Error"
    result = llm_decide(llm_name, output_text_data, error_text_data)
    print(result)
