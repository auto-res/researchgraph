import streamlit as st

from researchgraph.generator_subgraph.generator_subgraph import GeneratorSubgraph
from researchgraph.generator_subgraph.input_data import generator_subgraph_input_data
from researchgraph.executor_subgraph.executor_subgraph import ExecutorSubgraph
from researchgraph.writer_subgraph.writer_subgraph import WriterSubgraph
from researchgraph.upload_subgraph.upload_subgraph import UploadSubgraph


st.markdown("# Research Graph")
st.markdown("Research Graphは，機械学習研究の完全な自動化目的に開発しています．")

with st.sidebar:
    st.markdown("## AutoRes")
    st.markdown("https://www.autores.one/japanese")
    st.markdown("## Research Graph")
    st.markdown("https://github.com/auto-res/researchgraph")


repository = st.text_input(
    "新規手法の実装を管理するGitHubリポジトリ", value="auto-res2/auto-research"
)
github_owner, repository_name = repository.split("/", 1)
st.markdown("- https://github.com/auto-res2/auto-research")

on = st.toggle("Research Graphのアーキテクチャ")
if on:
    st.image("/workspaces/researchgraph/images/research_graph.png")

(ret, gen, exec, write, up) = st.tabs(
    ["Retriever", "Generator", "Executor", "Writer", "Uploader"]
)

# 共通の設定項目
save_dir = "/workspaces/researchgraph/data"
latex_template_file_path = "/workspaces/researchgraph/data/latex/template.tex"
figures_dir = "/workspaces/researchgraph/images"
pdf_file_path = "/workspaces/researchgraph/data/test_output.pdf"


# Retriever Subgraph
ret.markdown("## Retriever")
ret.markdown("Retriever Subgraphは，研究論文を取得するためのサブグラフです．")
if ret.button("研究のための論文を取得"):
    ret.write("実行中")
    retriever_output = {}
else:
    ret.write("未実行")
    retriever_output = {}
ret.json(retriever_output)


# Generator Subgraph
gen.markdown("## Generator")
gen.markdown("Generator Subgraphは，新規手法を生成するためのサブグラフです．")

on = gen.toggle("Generator Subgraphのアーキテクチャ")
if on:
    gen.image("/workspaces/researchgraph/images/generator_subgraph.png")

# new_method = ""
# experiment_code = ""
# verification_policy = ""
# experiment_details = ""
if gen.button("新規手法の生成"):
    gen.write("実行中")
    generator_subgraph = GeneratorSubgraph().build_graph()
    with st.spinner("Wait for it...", show_time=True):
        for event in generator_subgraph.stream(
            generator_subgraph_input_data, stream_mode="updates"
        ):
            node_name = list(event.keys())[0]
            gen.markdown(f"{node_name}の実行結果")
            # if event[node_name].get("new_method", ""):
            #     st.session_state["new_method"] = event[node_name]["new_method"]
            if event[node_name].get("experiment_code", ""):
                st.session_state["experiment_code"] = event[node_name][
                    "experiment_code"
                ]
            if event[node_name].get("verification_policy", ""):
                st.session_state["verification_policy"] = event[node_name][
                    "verification_policy"
                ]
            if event[node_name].get("experiment_details", ""):
                st.session_state["experiment_details"] = event[node_name][
                    "experiment_details"
                ]
            gen.json(event[node_name], expanded=False)
else:
    gen.write("未実行")


# Executor Subgraph
exec.markdown("## Executor")
exec.markdown("Executor Subgraphは，新規手法の実験をするサブグラフです．")

on = exec.toggle("Executor Subgraphのアーキテクチャ")
if on:
    exec.image("/workspaces/researchgraph/images/executor_subgraph.png")

max_code_fix_iteration = exec.slider(
    label="コードの修正回数", min_value=0, max_value=10, value=3, step=1
)

output_text_data = ""
devin_url = ""
branch_name = ""
if exec.button("新規手法の実験"):
    exec.write("実行中")
    executor_subgraph_input_data = {
        "fix_iteration_count": 0,
        "new_method": generator_subgraph_input_data["new_method"],
        "experiment_code": st.session_state["experiment_code"],
    }
    executor_subgraph = ExecutorSubgraph(
        github_owner=github_owner,
        repository_name=repository_name,
        save_dir=save_dir,
        max_code_fix_iteration=max_code_fix_iteration,
    ).build_graph()
    with st.spinner("Wait for it...", show_time=True):
        for event in executor_subgraph.stream(
            executor_subgraph_input_data, stream_mode="updates"
        ):
            node_name = list(event.keys())[0]
            gen.markdown(f"{node_name}の実行結果")
            if event[node_name].get("devin_url", ""):
                st.session_state["devin_url"] = event[node_name]["devin_url"]
                exec.markdown(f"- Devin URL: {st.session_state['devin_url']}")
            if event[node_name].get("branch_name", ""):
                st.session_state["branch_name"] = event[node_name]["branch_name"]
            if event[node_name].get("output_text_data", ""):
                st.session_state["output_text_data"] = event[node_name][
                    "output_text_data"
                ]
            gen.json(event[node_name], expanded=False)
else:
    exec.write("未実行")


# Writer Subgraph
write.markdown("## Writer")
write.markdown("Writer Subgraphは，論文を執筆するためのサブグラフです．")

on = write.toggle("Writer Subgraphのアーキテクチャ")
if on:
    write.image("/workspaces/researchgraph/images/writer_subgraph.png")

llm_name = write.selectbox(
    "「論文執筆」「Texへの変換」に使用するLLM",
    ("gpt-4o-mini-2024-07-18", "gpt-4o-2024-11-20"),
)

# paper_content = {}
if write.button("論文の執筆"):
    write.write("実行中")
    writer_subgraph_input_data = {
        "new_method": st.session_state["new_method"],
        "verification_policy": st.session_state["verification_policy"],
        "experiment_details": st.session_state["experiment_details"],
        "experiment_code": st.session_state["experiment_code"],
        "output_text_data": st.session_state["output_text_data"],
    }
    writer_subgraph = WriterSubgraph(
        llm_name=llm_name,
        latex_template_file_path=latex_template_file_path,
        figures_dir=figures_dir,
        pdf_file_path=pdf_file_path,
    ).build_graph()
    with st.spinner("Wait for it...", show_time=True):
        for event in writer_subgraph.stream(
            writer_subgraph_input_data, stream_mode="updates"
        ):
            node_name = list(event.keys())[0]
            gen.markdown(f"{node_name}の実行結果")
            if event[node_name].get("paper_content", ""):
                st.session_state["paper_content"] = event[node_name]["paper_content"]
            gen.json(event[node_name], expanded=False)
else:
    write.write("未実行")


# Uploader Subgraph
up.markdown("## Uploader")
up.markdown("Uploader Subgraphは，論文をアップロードするためのサブグラフです．")

on = up.toggle("Uploader Subgraphのアーキテクチャ")
if on:
    up.image("/workspaces/researchgraph/images/upload_subgraph.png")

if up.button("GitHubに結果をアップロード"):
    up.write("実行中")
    upload_subgraph_input_data = {
        "paper_content": {
            "Title": st.session_state["paper_content"].get("Title", ""),
            "Abstract": st.session_state["paper_content"].get("Abstract", ""),
        },
        "branch_name": st.session_state["branch_name"],
        "devin_url": st.session_state["devin_url"],
    }
    upload_subgraph = UploadSubgraph(
        github_owner=github_owner,
        repository_name=repository_name,
        pdf_file_path=pdf_file_path,
    ).build_graph()
    upload_output = upload_subgraph.invoke(upload_subgraph_input_data)
    if upload_output["completion"]:
        up.markdown(
            f"- GitHub URL: https://github.com/{repository}/tree/{upload_output['branch_name']}"
        )
else:
    up.write("未実行")
