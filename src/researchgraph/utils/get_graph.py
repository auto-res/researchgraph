import os
from IPython.display import Image
from langgraph.graph.graph import CompiledGraph

from researchgraph.executor_subgraph.executor_subgraph import ExecutorSubgraph
from researchgraph.retrieve_paper_subgraph.retrieve_paper_subgraph import (
    RetrievePaperSubgraph,
)
from researchgraph.writer_subgraph.writer_subgraph import WriterSubgraph
from researchgraph.integrate_generator_subgraph.integrate_generator_subgraph import (
    IntegrateGeneratorSubgraph,
)
from researchgraph.deep_research_subgraph.deep_research_subgraph import (
    DeepResearchSubgraph,
)
from researchgraph.research_graph import ResearchGraph

IMAGE_SAVE_DIR = "/workspaces/researchgraph/images"


def make_image(
    graph: CompiledGraph, file_name: str, image_save_dir: str = IMAGE_SAVE_DIR
):
    image = Image(graph.get_graph().draw_mermaid_png())
    with open(os.path.join(image_save_dir, file_name), "wb") as f:
        f.write(image.data)


def print_mermaid(graph):
    print(graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    save_dir = "/workspaces/researchgraph/data"
    llm_name = "gpt-4o-mini-2024-07-18"
    latex_template_file_path = "/workspaces/researchgraph/data/latex/template.tex"
    figures_dir = "/workspaces/researchgraph/images"

    retrieve_paper_subgraph = RetrievePaperSubgraph(
        llm_name=llm_name,
        save_dir=save_dir,
    ).build_graph()

    deep_research_subgraph = DeepResearchSubgraph(
        breadth=3,
        depth=2,
    ).build_graph()

    integrate_generator_subgraph = IntegrateGeneratorSubgraph(
        llm_name=llm_name,
    ).build_graph()

    executor_subgraph = ExecutorSubgraph(
        max_fix_iteration=3,
    ).build_graph()

    writer_graph = WriterSubgraph(
        llm_name=llm_name,
        latex_template_file_path=latex_template_file_path,
        figures_dir=figures_dir,
    ).build_graph()

    research_graph = ResearchGraph(
        llm_name=llm_name,
        save_dir=save_dir,
        max_fix_iteration=3,
        latex_template_file_path=latex_template_file_path,
        figures_dir=figures_dir,
    ).build_graph()

    make_image(graph=retrieve_paper_subgraph, file_name="retrieve_paper_subgraph.png")
    make_image(graph=deep_research_subgraph, file_name="deep_research_subgraph.png")
    make_image(
        graph=integrate_generator_subgraph, file_name="integrate_generator_subgraph.png"
    )
    make_image(graph=executor_subgraph, file_name="executor_subgraph.png")
    make_image(graph=writer_graph, file_name="writer_subgraph.png")
    make_image(graph=research_graph, file_name="research_graph.png")
    # print_mermaid(research_graph)
    # print_mermaid(retrieve_paper_subgraph)
    print_mermaid(deep_research_subgraph)
    # print_mermaid(integrate_generator_subgraph)
    # print_mermaid(executor_subgraph)
    # print_mermaid(writer_graph)
