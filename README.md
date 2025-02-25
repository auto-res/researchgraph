# ResearchGraph

ResearchGraphは完全な研究の自動化を目的としたOSSになります．

## ResearchGraph
全てのサブグラフを実行します．(現在はDeep Research Subgraphを組み込めていません)

```python
python src/researchgraph/research_graph.py
```

<details>

<summary>Architecture</summary>

```mermaid
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
        __start__([<p>__start__</p>]):::first
        retrieve_paper_subgraph(retrieve_paper_subgraph)
        generate_subgraph(generate_subgraph)
        executor_subgraph(executor_subgraph)
        writer_subgraph(writer_subgraph)
        __end__([<p>__end__</p>]):::last
        __start__ --> retrieve_paper_subgraph;
        executor_subgraph --> writer_subgraph;
        generate_subgraph --> executor_subgraph;
        retrieve_paper_subgraph --> generate_subgraph;
        writer_subgraph --> __end__;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
```

</details>


## Retrieve paper subgraph
ベースにする研究論文とそれに追加する技術の論文を取得するためのサブグラフです．

```python
python src/researchgraph/retrieve_paper_subgraph/retrieve_paper_subgraph.py
```


<details>

<summary>Architecture</summary>

```mermaid
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
        __start__([<p>__start__</p>]):::first
        initialize_state(initialize_state)
        base_search_papers_node(base_search_papers_node)
        base_retrieve_arxiv_full_text_node(base_retrieve_arxiv_full_text_node)
        base_extract_github_urls_node(base_extract_github_urls_node)
        base_summarize_paper_node(base_summarize_paper_node)
        base_select_best_paper_node(base_select_best_paper_node)
        generate_queries_node(generate_queries_node)
        add_search_papers_node(add_search_papers_node)
        add_retrieve_arxiv_full_text_node(add_retrieve_arxiv_full_text_node)
        add_extract_github_urls_node(add_extract_github_urls_node)
        add_summarize_paper_node(add_summarize_paper_node)
        add_select_best_paper_node(add_select_best_paper_node)
        __end__([<p>__end__</p>]):::last
        __start__ --> initialize_state;
        add_retrieve_arxiv_full_text_node --> add_extract_github_urls_node;
        add_search_papers_node --> add_retrieve_arxiv_full_text_node;
        add_select_best_paper_node --> __end__;
        base_retrieve_arxiv_full_text_node --> base_extract_github_urls_node;
        base_search_papers_node --> base_retrieve_arxiv_full_text_node;
        base_select_best_paper_node --> generate_queries_node;
        generate_queries_node --> add_search_papers_node;
        initialize_state --> base_search_papers_node;
        base_extract_github_urls_node -. &nbsp;Next paper&nbsp; .-> base_retrieve_arxiv_full_text_node;
        base_extract_github_urls_node -. &nbsp;Generate paper summary&nbsp; .-> base_summarize_paper_node;
        base_extract_github_urls_node -. &nbsp;All complete&nbsp; .-> base_select_best_paper_node;
        base_summarize_paper_node -. &nbsp;Next paper&nbsp; .-> base_retrieve_arxiv_full_text_node;
        base_summarize_paper_node -. &nbsp;All complete&nbsp; .-> base_select_best_paper_node;
        add_extract_github_urls_node -. &nbsp;Next paper&nbsp; .-> add_retrieve_arxiv_full_text_node;
        add_extract_github_urls_node -. &nbsp;Generate paper summary&nbsp; .-> add_summarize_paper_node;
        add_extract_github_urls_node -. &nbsp;All complete&nbsp; .-> add_select_best_paper_node;
        add_summarize_paper_node -. &nbsp;Next paper&nbsp; .-> add_retrieve_arxiv_full_text_node;
        add_summarize_paper_node -. &nbsp;All complete&nbsp; .-> add_select_best_paper_node;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
```
</details>


## Deep Research Subgraph
Web上から情報を取得するためのサブグラフです．

```python
python src/researchgraph/deep_research_subgraph/deep_research_subgraph.py
```


<details>

<summary>Architecture</summary>

```mermaid
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
        __start__([<p>__start__</p>]):::first
        recursive_search_node(recursive_search_node)
        generate_report_node(generate_report_node)
        __end__([<p>__end__</p>]):::last
        __start__ --> recursive_search_node;
        generate_report_node --> __end__;
        recursive_search_node --> generate_report_node;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
```
</details>


## Integrate Generator Subgraph
手法を合成するためのサブグラフです．

```python
python src/researchgraph/integrate_generator_subgraph/integrate_generator_subgraph.py
```


<details>

<summary>Architecture</summary>

```mermaid
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
        __start__([<p>__start__</p>]):::first
        retrieve_base_paper_code_with_devin(retrieve_base_paper_code_with_devin)
        retrieve_add_paper_code_with_devin(retrieve_add_paper_code_with_devin)
        method_integrate_node(method_integrate_node)
        __end__([<p>__end__</p>]):::last
        __start__ --> retrieve_add_paper_code_with_devin;
        __start__ --> retrieve_base_paper_code_with_devin;
        method_integrate_node --> __end__;
        retrieve_add_paper_code_with_devin --> method_integrate_node;
        retrieve_base_paper_code_with_devin --> method_integrate_node;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
```
</details>


## Executor Subgraph
新規の手法をコーディングし実行するためのサブグラフです．

```python
python src/researchgraph/executor_subgraph/executor_subgraph.py
```


<details>

<summary>Architecture</summary>

```mermaid
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
        __start__([<p>__start__</p>]):::first
        generate_code_with_devin_node(generate_code_with_devin_node)
        execute_github_actions_workflow_node(execute_github_actions_workflow_node)
        retrieve_github_actions_artifacts_node(retrieve_github_actions_artifacts_node)
        fix_code_with_devin_node(fix_code_with_devin_node)
        __end__([<p>__end__</p>]):::last
        __start__ --> generate_code_with_devin_node;
        execute_github_actions_workflow_node --> retrieve_github_actions_artifacts_node;
        fix_code_with_devin_node --> execute_github_actions_workflow_node;
        generate_code_with_devin_node --> execute_github_actions_workflow_node;
        retrieve_github_actions_artifacts_node -. &nbsp;correction&nbsp; .-> fix_code_with_devin_node;
        retrieve_github_actions_artifacts_node -. &nbsp;finish&nbsp; .-> __end__;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
```
</details>


## Writer Subgraph
論文を執筆するためのサブグラフです．執筆した論文はGitHub上にアップロードされます．

```python
python src/researchgraph/writer_subgraph/writer_subgraph.py
```


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
