from datetime import datetime
from typing import List
from litellm import completion
from jinja2 import Environment
from pydantic import BaseModel
import json


class ReportSummary(BaseModel):
    executive_summary: str


class DetailedFindings(BaseModel):
    detailed_findings: str


def generate_report(query: str, learnings: List[str], visited_urls: List[str]) -> str:
    learning_str = "\n".join(f"- {learning}" for learning in learnings)
    sections = [
        create_header(query),
        create_summary(query, learning_str),
        create_findings(query, learning_str),
        create_sources(visited_urls),
    ]
    return "\n\n".join(sections)


def create_header(query: str) -> str:
    """レポートのヘッダー部分を作成"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"# Research Report\n"
        f"Generated on: {timestamp}\n\n"
        f"## Original Query\n"
        f"{query}"
    )


def create_summary(query: str, learning_str: str) -> str:
    data = {"learning_str": learning_str}
    prompt_template = """
Please create a summary of the answers to the questions listed in the “Question” section based on the information in the “Knowledge” section.
Please do not use any information other than that provided by “Knowledge”.
# Question
{{ query }}
# Knowledge
{{ learning_str }}"""
    env = Environment()
    template = env.from_string(prompt_template)
    prompt = template.render(data)
    response = completion(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "user", "content": f"{prompt}"},
        ],
        response_format=ReportSummary,
    )
    output = response.choices[0].message.content
    output_dict = json.loads(output)
    executive_summary = f"## Executive Summary\n{output_dict['executive_summary']}"
    return executive_summary


def create_findings(query: str, learning_str: str) -> str:
    data = {"learning_str": learning_str}
    prompt_template = """
Please create a detailed response to the information in the “Question” section, using the information in the “Knowledge” section as a guide.
Please do not use any information other than that provided by “Knowledge”.
# Question
{{ query }}
# Knowledge
{{ learning_str }}"""
    env = Environment()
    template = env.from_string(prompt_template)
    prompt = template.render(data)
    response = completion(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "user", "content": f"{prompt}"},
        ],
        response_format=DetailedFindings,
    )
    output = response.choices[0].message.content
    output_dict = json.loads(output)
    detailed_findings = f"## Detailed Findings\n{output_dict['detailed_findings']}"
    return detailed_findings


def create_sources(urls: List[str]) -> str:
    """リサーチで参照した情報源（URLリスト）を作成"""
    if not urls:
        return "## Sources\nNo sources to cite."

    sources = "\n".join(f"- {url}" for url in urls)
    return f"## Sources\n{sources}"


# 実行例
if __name__ == "__main__":
    query = "How does deep learning impact medical research?"
    learnings = [
        "Deep learning improves diagnostic accuracy in medical imaging.",
        "Neural networks help in drug discovery by predicting molecular interactions.",
        "AI models assist doctors in predicting patient outcomes with high precision.",
    ]
    visited_urls = [
        "https://www.ncbi.nlm.nih.gov/",
        "https://arxiv.org/",
        "https://www.nature.com/",
    ]

    report = generate_report(
        query=query, learnings=learnings, visited_urls=visited_urls
    )
    print(report)
