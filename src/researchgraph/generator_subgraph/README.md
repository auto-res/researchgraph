# Generator Subgraph  
新規手法を生成するためのサブグラフです．

  <details>

  <summary>Architecture</summary>

  ```mermaid
  %%{init: {'flowchart': {'curve': 'linear'}}}%%
  graph TD;
          __start__([<p>__start__</p>]):::first
          generator_node(generator_node)
          __end__([<p>__end__</p>]):::last
          __start__ --> generator_node;
          generator_node --> __end__;
          classDef default fill:#f2f0ff,line-height:1.2
          classDef first fill-opacity:0
          classDef last fill:#bfb6fc
  ```
  </details>


## How to execute

```python
uv run python src/researchgraph/generator_subgraph/generator_subgraph.py
```
