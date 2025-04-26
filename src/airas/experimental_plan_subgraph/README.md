# Experimental Plan Subgraph  
実験計画をたて，コーディングを行うためのサブグラフです．

<details>

<summary>Architecture</summary>

```mermaid
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
        __start__([<p>__start__</p>]):::first
        retrieve_code_with_devin_node(retrieve_code_with_devin_node)
        check_devin_completion_node(check_devin_completion_node)
        generate_advantage_criteria_node(generate_advantage_criteria_node)
        generate_experiment_details_node(generate_experiment_details_node)
        generate_experiment_code_node(generate_experiment_code_node)
        __end__([<p>__end__</p>]):::last
        __start__ --> generate_advantage_criteria_node;
        __start__ --> retrieve_code_with_devin_node;
        check_devin_completion_node --> generate_experiment_details_node;
        generate_advantage_criteria_node --> generate_experiment_details_node;
        generate_experiment_code_node --> __end__;
        generate_experiment_details_node --> generate_experiment_code_node;
        retrieve_code_with_devin_node --> check_devin_completion_node;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
```
</details>


## How to execute

```python
uv run python /workspaces/researchgraph/src/researchgraph/experimental_plan_subgraph/experimental_plan_subgraph.py
```
