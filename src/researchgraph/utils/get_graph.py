import os
from IPython.display import Image
from langgraph.graph.graph import CompiledGraph

from researchgraph.executor_subgraph.executor_subgraph import ExecutorSubgraph

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
    # image_save_dir = "/workspaces/researchgraph/images"
    # save_dir = "/workspaces/researchgraph/data"
    # llm_name = "gpt-4o-mini-2024-07-18"

    # graph = RetrievePaperSubgraph(
    #     llm_name=llm_name,
    #     save_dir=save_dir,
    # ).build_graph()
    # file_name = "retrieve_paper_subgraph.png"

    # llm_name = "gpt-4o-2024-11-20"
    # graph = IntegrateGeneratorSubgraph(
    #     llm_name=llm_name,
    #     ai_integrator_v3_creator_prompt=integrate_generator_subgraph.ai_integrator_v3_creator_prompt,
    # ).build_graph()
    # file_name = "integrate_generator_subgraph.png"

    graph = ExecutorSubgraph(
        max_fix_iteration=3,
    ).build_graph()
    file_name = "executor_subgraph.png"

    # latex_template_file_path = "/workspaces/researchgraph/data/latex/template.tex"
    # figures_dir = "/workspaces/researchgraph/images"
    # llm_name = "gpt-4o-mini-2024-07-18"

    # graph = WriterSubgraph(
    #     llm_name=llm_name,
    #     latex_template_file_path=latex_template_file_path,
    #     figures_dir=figures_dir,
    # ).build_graph()
    # file_name = "writer_subgraph.png"

    make_image(
        graph=graph,
        file_name=file_name,
    )
    # print_mermaid(graph)
