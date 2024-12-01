from litellm import completion
import ast


def litellm_output(llm_name: str, prompt: str, response_field) -> dict:
    response = completion(
        model=llm_name,
        messages=[
            {"role": "user", "content": prompt},
        ],
        response_format=response_field,
    )
    output = response.choices[0].message.content
    output_dict = ast.literal_eval(output)
    return output_dict
