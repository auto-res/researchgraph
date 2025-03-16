import streamlit as st

st.markdown("# Research Graph")
st.markdown("Research Graphは，機械学習研究の完全な自動化目的に開発しています．")

with st.sidebar:
    st.markdown("## AutoRes")
    st.markdown("https://www.autores.one/japanese")
    st.markdown("## Research Graph")
    st.markdown("https://github.com/auto-res/researchgraph")


(ret, gen, exec, write, up) = st.tabs(
    ["Retriever", "Generator", "Executor", "Writer", "Uploader"]
)


ret.markdown("## Retriever")
ret.markdown("Retriever Subgraphは，研究論文を取得するためのサブグラフです．")


gen.markdown("## Generator")
gen.markdown("Generator Subgraphは，新規手法を生成するためのサブグラフです．")


exec.markdown("## Executor")
exec.markdown("Executor Subgraphは，新規手法の実験をするサブグラフです．")

write.markdown("## Writer")
write.markdown("Writer Subgraphは，論文を執筆するためのサブグラフです．")

up.markdown("## Uploader")
up.markdown("Uploader Subgraphは，論文をアップロードするためのサブグラフです．")
