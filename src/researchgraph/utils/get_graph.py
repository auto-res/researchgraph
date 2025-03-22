import os
from IPython.display import Image
from langgraph.graph.graph import CompiledGraph

from researchgraph.research_graph import ResearchGraph

from researchgraph.retrieve_paper_subgraph.retrieve_paper_subgraph import (
    RetrievePaperSubgraph,
)
from researchgraph.generator_subgraph.generator_subgraph import GeneratorSubgraph
from researchgraph.experimental_plan_subgraph.experimental_plan_subgraph import (
    ExperimentalPlanSubgraph,
)
from researchgraph.executor_subgraph.executor_subgraph import ExecutorSubgraph
from researchgraph.writer_subgraph.writer_subgraph import WriterSubgraph
from researchgraph.upload_subgraph.upload_subgraph import UploadSubgraph


IMAGE_SAVE_DIR = "/workspaces/researchgraph/images"


def make_image(
    graph: CompiledGraph, file_name: str, image_save_dir: str = IMAGE_SAVE_DIR
):
    image = Image(graph.get_graph(xray=2).draw_mermaid_png())
    with open(os.path.join(image_save_dir, file_name), "wb") as f:
        f.write(image.data)


def print_mermaid(graph):
    print(graph.get_graph(xray=2).draw_mermaid())


if __name__ == "__main__":
    save_dir = "/workspaces/researchgraph/data"
    llm_name = "gpt-4o-mini-2024-07-18"
    latex_template_file_path = "/workspaces/researchgraph/data/latex/template.tex"
    figures_dir = "/workspaces/researchgraph/images"
    scrape_urls = [
        "https://icml.cc/virtual/2024/papers.html?filter=titles",
        "https://iclr.cc/virtual/2024/papers.html?filter=titles",
        # "https://nips.cc/virtual/2024/papers.html?filter=titles",
        # "https://cvpr.thecvf.com/virtual/2024/papers.html?filter=titles",
    ]

    retrieve_paper_subgraph = RetrievePaperSubgraph(
        llm_name=llm_name,
        save_dir=save_dir,
        scrape_urls=scrape_urls,
        add_paper_num=3,
    ).build_graph()

    generator_subgraph = GeneratorSubgraph().build_graph()

    experimental_plan_subgraph = ExperimentalPlanSubgraph().build_graph()

    executor_subgraph = ExecutorSubgraph(
        github_owner="auto-res2",
        repository_name="auto-research",
        save_dir=save_dir,
        max_code_fix_iteration=3,
    ).build_graph()

    writer_graph = WriterSubgraph(
        llm_name=llm_name,
        latex_template_file_path=latex_template_file_path,
        figures_dir=figures_dir,
        pdf_file_path="/workspaces/researchgraph/data/test_output.pdf",
    ).build_graph()

    upload_subgraph = UploadSubgraph(
        github_owner="auto-res2",
        repository_name="auto-research",
        pdf_file_path="/workspaces/researchgraph/data/test_output.pdf",
    ).build_graph()

    research_graph = ResearchGraph(
        llm_name=llm_name,
        save_dir=save_dir,
        scrape_urls=scrape_urls,
        add_paper_num=3,
        max_code_fix_iteration=3,
        github_owner="auto-res2",
        repository_name="auto-research",
        pdf_file_path="/workspaces/researchgraph/data/test_output.pdf",
        latex_template_file_path=latex_template_file_path,
        figures_dir=figures_dir,
    ).build_graph()

    # make_image(graph=retrieve_paper_subgraph, file_name="retrieve_paper_subgraph.png")
    # make_image(graph=generator_subgraph, file_name="generator_subgraph.png")
    # make_image(graph=experimental_plan_subgraph, file_name="experimental_plan_subgraph.png")
    # make_image(graph=executor_subgraph, file_name="executor_subgraph.png")
    # make_image(graph=writer_graph, file_name="writer_subgraph.png")
    # make_image(graph=upload_subgraph, file_name="upload_subgraph.png")
    # make_image(graph=research_graph, file_name="research_graph.png")
    # print_mermaid(research_graph)
    # print_mermaid(retrieve_paper_subgraph)
    print_mermaid(generator_subgraph)
    # print_mermaid(experimental_plan_subgraph)
    # print_mermaid(retrieve_paper_subgraph)
    # print_mermaid(executor_subgraph)
    # print_mermaid(writer_graph)
