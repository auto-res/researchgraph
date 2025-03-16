# ResearchGraph
ResearchGraph is created by [AutoRes](https://www.autores.one/english).
ResearchGraph is an OSS that aims to automate complete machine learning research and to self-improve the automatic research system.  
ResearchGraphは[AutoRes](https://www.autores.one/japanese)というプロジェクトで作成しています．
ResearchGraphは完全な機械学習研究の自動化および，自動研究システムの自己改善を目的としたOSSになります．

## Explanation


### ResearchGraph Architecture

<details>

<summary>Architecture</summary>

```mermaid
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
        __start__([<p>__start__</p>]):::first
        generator_subgraph_generate_advantage_criteria_node(generate_advantage_criteria_node)
        generator_subgraph_generate_experiment_details_node(generate_experiment_details_node)
        generator_subgraph_generate_experiment_code_node(generate_experiment_code_node)
        executor_subgraph_generate_code_with_devin_node(generate_code_with_devin_node)
        executor_subgraph_execute_github_actions_workflow_node(execute_github_actions_workflow_node)
        executor_subgraph_retrieve_github_actions_artifacts_node(retrieve_github_actions_artifacts_node)
        executor_subgraph_llm_decide_node(llm_decide_node)
        executor_subgraph_fix_code_with_devin_node(fix_code_with_devin_node)
        executor_subgraph___end__(<p>__end__</p>)
        writer_subgraph_writeup_node(writeup_node)
        writer_subgraph_latex_node(latex_node)
        __end__([<p>__end__</p>]):::last
        __start__ --> generator_subgraph_generate_advantage_criteria_node;
        executor_subgraph___end__ --> writer_subgraph_writeup_node;
        generator_subgraph_generate_experiment_code_node --> executor_subgraph_generate_code_with_devin_node;
        writer_subgraph_latex_node --> __end__;
        subgraph generator_subgraph
        generator_subgraph_generate_advantage_criteria_node --> generator_subgraph_generate_experiment_details_node;
        generator_subgraph_generate_experiment_details_node --> generator_subgraph_generate_experiment_code_node;
        end
        subgraph executor_subgraph
        executor_subgraph_execute_github_actions_workflow_node --> executor_subgraph_retrieve_github_actions_artifacts_node;
        executor_subgraph_fix_code_with_devin_node --> executor_subgraph_execute_github_actions_workflow_node;
        executor_subgraph_generate_code_with_devin_node --> executor_subgraph_execute_github_actions_workflow_node;
        executor_subgraph_retrieve_github_actions_artifacts_node --> executor_subgraph_llm_decide_node;
        executor_subgraph_llm_decide_node -. &nbsp;correction&nbsp; .-> executor_subgraph_fix_code_with_devin_node;
        executor_subgraph_llm_decide_node -. &nbsp;finish&nbsp; .-> executor_subgraph___end__;
        end
        subgraph writer_subgraph
        writer_subgraph_writeup_node --> writer_subgraph_latex_node;
        end
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
```

</details>



- Deep Research Subgraph
Web上から情報を取得するためのサブグラフです．

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


- Generator Subgraph
手法を合成するためのサブグラフです．

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


- Executor Subgraph
新規の手法をコーディングし実行するためのサブグラフです．

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


- Writer Subgraph
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


### Result
The following is a repository that summarizes the results of ResearchGraph.
- [auto-research](https://github.com/auto-res2/auto-research)



## Settings


## How to execute
- Research Graph
```python
python src/researchgraph/research_graph.py
```
- Deep Research Subgraph
```python
python src/researchgraph/deep_research_subgraph/deep_research_subgraph.py
```
- Generate Subgraph
```python
python src/researchgraph/integrate_generator_subgraph/integrate_generator_subgraph.py
```
- Executor Subgraph
```python
python src/researchgraph/executor_subgraph/executor_subgraph.py
```
- writer subgraph
```python
python src/researchgraph/writer_subgraph/writer_subgraph.py
```
