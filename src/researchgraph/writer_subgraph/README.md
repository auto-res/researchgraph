# Writer Subgraph  
論文を執筆するためのサブグラフです．執筆した論文はGitHub上にアップロードされます．

<details>

<summary>Architecture</summary>

```mermaid
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
        __start__([<p>__start__</p>]):::first
        writeup_node(writeup_node)
        latex_node(latex_node)
        github_upload_node(github_upload_node)
        __end__([<p>__end__</p>]):::last
        __start__ --> writeup_node;
        github_upload_node --> __end__;
        latex_node --> github_upload_node;
        writeup_node --> latex_node;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
```
</details>


## How to execute

```python
uv run python src/researchgraph/writer_subgraph/writer_subgraph.py
```
