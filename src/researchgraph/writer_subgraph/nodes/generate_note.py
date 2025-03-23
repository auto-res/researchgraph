from jinja2 import Template


corresponding_section: dict[str, list[str]] = {
    "Title": [],
    "Methods": [
        "base_method_text",
        "new_method",
        "verification_policy",
        "experiment_details",
    ],
    "Codes": ["experiment_code"],
    "Results": ["output_text_data"],
    "Analysis": [],
    # "Related Work":[]
}


def generate_note(state: dict) -> str:
    template = Template("""
    {% for section, items in sections.items() %}
    # {{ section }}
    {% for key, value in items.items() %}
    {{ key }}: {{ value }}
    {% endfor %}
    {% endfor %}
    """)

    sections: dict[str, dict] = {}
    for section, state_name_list in corresponding_section.items():
        matched_items: dict[str, str] = {}
        for state_name in state_name_list:
            matched_items[state_name] = state[state_name]
        sections[section] = matched_items
    return template.render(sections=sections)
