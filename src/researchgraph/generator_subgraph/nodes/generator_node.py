from litellm import completion


def generator_node(
    base_method_text: str,
    add_method_text_list: list[str],
) -> str:
    add_method_text = "---".join(add_method_text_list)
    prompt = f"""
Please create a new research method based on the ideas of some of the methods given in the “Add Method” using the research methods given in the “Base Method”.
# Base Method
{base_method_text}

# Add Method
{add_method_text}
"""

    response = completion(
        model="o3-mini-2025-01-31",
        messages=[
            {"role": "user", "content": f"{prompt}"},
        ],
    )
    new_method = response.choices[0].message.content

    return new_method
