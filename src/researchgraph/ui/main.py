import streamlit as st

import sys

sys.path.append("/mount/src/researchgraph/src")

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


st.markdown("# Research Graph")
st.markdown("Research Graphは，機械学習研究の完全な自動化目的に開発しています．")

with st.sidebar:
    st.markdown("## AutoRes")
    st.markdown("https://www.autores.one/japanese")
    st.markdown("## Research Graph")
    st.markdown("https://github.com/auto-res/researchgraph")


st.markdown("[実行結果一覧](https://github.com/auto-res2/auto-research)")


# 共通の設定項目
save_dir = "/tmp/data"
scrape_urls = [
    "https://icml.cc/virtual/2024/papers.html?filter=titles",
    "https://iclr.cc/virtual/2024/papers.html?filter=titles",
    # "https://nips.cc/virtual/2024/papers.html?filter=titles",
    # "https://cvpr.thecvf.com/virtual/2024/papers.html?filter=titles",
]


st.markdown("- アーキテクチャ")
on = st.toggle("表示", key="research_graph_architecture")
if on:
    st.image("images/research_graph.png")

st.markdown("- 設定")
st.session_state["repository"] = st.text_input(
    "新規手法の実装を管理するGitHubリポジトリを設定してください．",
    value="auto-res2/auto-research",
)
st.session_state["github_owner"], st.session_state["repository_name"] = (
    st.session_state["repository"].split("/", 1)
)
st.session_state["query"] = st.text_input(
    "何の研究をしますか？", "diffusion model", key="research_graph_query"
)
st.session_state["research_graph_input_data"] = {
    "queries": [st.session_state["query"]],
}
st.session_state["add_paper_num"] = st.slider(
    label="ベース論文をアップデートする際に使用する論文数",
    min_value=1,
    max_value=15,
    value=3,
    step=1,
    key="research_graph_add_paper_num",
)
st.session_state["max_code_fix_iteration"] = st.slider(
    label="コードの修正回数",
    min_value=0,
    max_value=10,
    value=3,
    step=1,
    key="research_graph_max_code_fix_iteration",
)

st.markdown("- 実行")
if st.button("start", key="research_graph_start"):
    st.write("実行中")
    research_graph = ResearchGraph(
        save_dir=save_dir,
        scrape_urls=scrape_urls,
        add_paper_num=st.session_state["add_paper_num"],
        repository=st.session_state["repository"],
        max_code_fix_iteration=st.session_state["max_code_fix_iteration"],
    ).build_graph()
    with st.spinner("Wait for it...", show_time=True):
        for event in research_graph.stream(
            st.session_state["research_graph_input_data"],
            stream_mode="updates",
            config={"recursion_limit": 500},
        ):
            subgraph_name = list(event.keys())[0]
            # Execute Subgraph
            if event[subgraph_name].get("experiment_devin_url", ""):
                st.session_state["experiment_devin_url"] = event[subgraph_name][
                    "experiment_devin_url"
                ]
                st.markdown(
                    f"- Experiment Devin URL: {st.session_state['experiment_devin_url']}"
                )
            elif event[subgraph_name].get("branch_name", ""):
                st.session_state["branch_name"] = event[subgraph_name]["branch_name"]
            # Uploader Subgraph
            elif event[subgraph_name].get("completion", ""):
                st.markdown(
                    "自動実験が完了しました．実行結果は以下のリンクから確認できます．"
                )
                st.markdown(
                    f"- GitHub URL: https://github.com/{st.session_state['repository_name']}/tree/{st.session_state['branch_name']}"
                )
            st.markdown(f"{subgraph_name}の実行結果")
            st.json(event[subgraph_name], expanded=False)
else:
    st.write("未実行")


st.markdown(
    "______________________________________________________________________________________"
)
st.markdown("### Subgraph単位での実行")
(ret, gen, expe, exec, write, up) = st.tabs(
    [
        "Retrieve Paper",
        "Generator",
        "Experimental Plan",
        "Executor",
        "Writer",
        "Uploader",
    ]
)


# Retrieve Paper Subgraph
ret.markdown("#### Retrieve Paper Subgraph")
ret.markdown("Retrieve Paper Subgraphは，研究論文を取得するためのサブグラフです．")


ret.markdown("- アーキテクチャ")
on = ret.toggle("表示", key="retrieve_paper_subgraph_architecture")
if on:
    ret.image("images/retrieve_paper_subgraph.png")


ret.markdown("- 設定")
st.session_state["query"] = ret.text_input(
    "何の研究をしますか？", "diffusion model", key="retrieve_paper_subgraph_query"
)
st.session_state["retrieve_paper_subgraph_input_data"] = {
    "queries": [st.session_state["query"]],
}
st.session_state["add_paper_num"] = ret.slider(
    label="ベース論文をアップデートする際に使用する論文数",
    min_value=1,
    max_value=15,
    value=3,
    step=1,
    key="retrieve_paper_subgraph_add_paper_num",
)


ret.markdown("- 実行")
if ret.button("start", key="retrieve_paper_subgraph_start"):
    ret.write("実行中")
    retrieve_paper_subgraph = RetrievePaperSubgraph(
        llm_name="gpt-4o-mini-2024-07-18",
        save_dir=save_dir,
        scrape_urls=scrape_urls,
        add_paper_num=st.session_state["add_paper_num"],
    ).build_graph()
    with st.spinner("Wait for it...", show_time=True):
        for event in retrieve_paper_subgraph.stream(
            st.session_state["retrieve_paper_subgraph_input_data"],
            stream_mode="updates",
            config={"recursion_limit": 500},
        ):
            node_name = list(event.keys())[0]
            if event[node_name].get("base_github_url", ""):
                st.session_state["base_github_url"] = event[node_name][
                    "base_github_url"
                ]
            elif event[node_name].get("base_method_text", ""):
                st.session_state["base_method_text"] = event[node_name][
                    "base_method_text"
                ]
            elif event[node_name].get("add_method_texts", ""):
                st.session_state["add_method_texts"] = event[node_name][
                    "add_method_texts"
                ]
            ret.markdown(f"{node_name}の実行結果")
            ret.json(event[node_name], expanded=False)
else:
    ret.write("未実行")


# Generator Subgraph
gen.markdown("#### Generator Subgraph")
gen.markdown("Generator Subgraphは，新規手法を生成するためのサブグラフです．")


gen.markdown("- アーキテクチャ")
on = gen.toggle("表示", key="generator_subgraph_architecture")
if on:
    gen.image("images/generator_subgraph.png")


gen.markdown("- 実行")
if gen.button("start", key="generator_subgraph_start"):
    gen.write("実行中")
    generator_subgraph_input_data = {
        "base_method_text": st.session_state["base_method_text"],
        "add_method_texts": st.session_state["add_method_texts"],
    }
    generator_subgraph = GeneratorSubgraph().build_graph()
    with st.spinner("Wait for it...", show_time=True):
        for event in generator_subgraph.stream(
            generator_subgraph_input_data, stream_mode="updates"
        ):
            node_name = list(event.keys())[0]
            if event[node_name].get("new_method", ""):
                st.session_state["new_method"] = event[node_name]["new_method"]
            gen.markdown(f"{node_name}の実行結果")
            gen.json(event[node_name], expanded=False)
else:
    gen.write("未実行")


# Experimental Plan Subgraph
expe.markdown("#### Experimental Plan Subgraph")
expe.markdown(
    "Experimental Plan Subgraphは，実験の計画および実験スクリプトを作成するめのサブグラフです．"
)


expe.markdown("- アーキテクチャ")
on = expe.toggle("表示", key="experimental_plan_subgraph_architecture")
if on:
    expe.image("images/experimental_plan_subgraph.png")


expe.markdown("- 実行")
if expe.button("start", key="experimental_plan_subgraph_start"):
    expe.write("実行中")
    experimental_plan_subgraph_input_data = {
        "new_method": st.session_state["new_method"],
        "base_github_url": st.session_state["base_github_url"],
        "base_method_text": st.session_state["base_method_text"],
    }
    experimental_plan_subgraph = ExperimentalPlanSubgraph().build_graph()
    with st.spinner("Wait for it...", show_time=True):
        for event in experimental_plan_subgraph.stream(
            experimental_plan_subgraph_input_data,
            stream_mode="updates",
        ):
            node_name = list(event.keys())[0]
            expe.markdown(f"{node_name}の実行結果")
            if event[node_name].get("experiment_code", ""):
                st.session_state["experiment_code"] = event[node_name][
                    "experiment_code"
                ]
            elif event[node_name].get("verification_policy", ""):
                st.session_state["verification_policy"] = event[node_name][
                    "verification_policy"
                ]
            elif event[node_name].get("experiment_details", ""):
                st.session_state["experiment_details"] = event[node_name][
                    "experiment_details"
                ]
            elif event[node_name].get("retrieve_devin_url", ""):
                st.session_state["retrieve_devin_url"] = event[node_name][
                    "retrieve_devin_url"
                ]
                expe.markdown(
                    f"- Retrieve Devin URL: {st.session_state['retrieve_devin_url']}"
                )
            expe.json(event[node_name], expanded=False)
else:
    expe.write("未実行")


# Executor Subgraph
exec.markdown("#### Executor Subgraph")
exec.markdown("Executor Subgraphは，新規手法の実験をするサブグラフです．")


exec.markdown("- アーキテクチャ")
on = exec.toggle("表示", key="executor_subgraph_architecture")
if on:
    exec.image(
        "images/executor_subgraph.png",
        caption="Web上の画像",
    )


exec.markdown("- 設定")
repository = exec.text_input(
    "新規手法の実装を管理するGitHubリポジトリを設定", value="auto-res2/auto-research"
)
github_owner, repository_name = repository.split("/", 1)
max_code_fix_iteration = exec.slider(
    label="コードの修正回数",
    min_value=0,
    max_value=10,
    value=3,
    step=1,
    key="executor_subgraph_max_code_fix_iteration",
)


exec.markdown("- 実行")
if exec.button("start", key="executor_subgraph_start"):
    exec.write("実行中")
    executor_subgraph_input_data = {
        "fix_iteration_count": 0,
        "new_method": st.session_state["new_method"],
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
            if event[node_name].get("experiment_devin_url", ""):
                st.session_state["experiment_devin_url"] = event[node_name][
                    "experiment_devin_url"
                ]
                exec.markdown(
                    f"- Experiment Devin URL: {st.session_state['experiment_devin_url']}"
                )
            elif event[node_name].get("branch_name", ""):
                st.session_state["branch_name"] = event[node_name]["branch_name"]
            elif event[node_name].get("output_text_data", ""):
                st.session_state["output_text_data"] = event[node_name][
                    "output_text_data"
                ]
            elif event[node_name].get("execution_logs", ""):
                st.session_state["execution_logs"] = event[node_name]["execution_logs"]
            gen.json(event[node_name], expanded=False)
else:
    exec.write("未実行")


# Writer Subgraph
write.markdown("#### Writer Subgraph")
write.markdown("Writer Subgraphは，論文を執筆するためのサブグラフです．")

write.markdown("- アーキテクチャ")
on = write.toggle("表示", key="writer_subgraph_architecture")
if on:
    write.image("images/writer_subgraph.png")

write.markdown("- 設定")
llm_name = write.selectbox(
    "「論文執筆」「Texへの変換」に使用するLLM",
    ("gpt-4o-mini-2024-07-18", "gpt-4o-2024-11-20"),
)

write.markdown("- 実行")
if write.button("start", key="writer_subgraph_start"):
    write.write("実行中")
    writer_subgraph_input_data = {
        "base_method_text": st.session_state["base_method_text"],
        "new_method": st.session_state["new_method"],
        "verification_policy": st.session_state["verification_policy"],
        "experiment_details": st.session_state["experiment_details"],
        "experiment_code": st.session_state["experiment_code"],
        "output_text_data": st.session_state["output_text_data"],
    }
    writer_subgraph = WriterSubgraph(
        llm_name=llm_name,
        save_dir=save_dir,
    ).build_graph()
    with st.spinner("Wait for it...", show_time=True):
        for event in writer_subgraph.stream(
            writer_subgraph_input_data, stream_mode="updates"
        ):
            node_name = list(event.keys())[0]
            write.markdown(f"{node_name}の実行結果")
            if event[node_name].get("paper_content", ""):
                st.session_state["paper_content"] = event[node_name]["paper_content"]
            write.json(event[node_name], expanded=False)
else:
    write.write("未実行")


# Uploader Subgraph
up.markdown("#### Uploader Subgraph")
up.markdown("Uploader Subgraphは，論文をアップロードするためのサブグラフです．")

up.markdown("- アーキテクチャ")
on = up.toggle("表示", key="uploader_subgraph_architecture")
if on:
    up.image("images/upload_subgraph.png")

up.markdown("- 実行")
if up.button("start", key="uploader_subgraph_start"):
    up.write("実行中")
    upload_subgraph_input_data = {
        "paper_content": {
            "Title": st.session_state["paper_content"].get("Title", ""),
            "Abstract": st.session_state["paper_content"].get("Abstract", ""),
        },
        "branch_name": st.session_state["branch_name"],
        "output_text_data": st.session_state["output_text_data"],
        "experiment_devin_url": st.session_state["devin_url"],
        "base_method_text": st.session_state["base_method_text"],
        "execution_logs": st.session_state["execution_logs"],
    }
    upload_subgraph = UploadSubgraph(
        github_owner=github_owner,
        repository_name=repository_name,
        save_dir=save_dir,
    ).build_graph()
    upload_output = upload_subgraph.invoke(upload_subgraph_input_data)
    if upload_output["completion"]:
        up.markdown(
            f"- GitHub URL: https://github.com/{repository}/tree/{upload_output['branch_name']}"
        )
else:
    up.write("未実行")
