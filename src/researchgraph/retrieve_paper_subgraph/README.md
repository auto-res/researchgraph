
# Retriever Subgraph  
ベースにする研究論文を取得するためのサブグラフです．

<details>

<summary>Architecture</summary>

```mermaid
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
        __start__([<p>__start__</p>]):::first
        initialize_state(initialize_state)
        base_web_scrape_node(base_web_scrape_node)
        base_extract_paper_title_node(base_extract_paper_title_node)
        base_search_arxiv_node(base_search_arxiv_node)
        base_retrieve_arxiv_full_text_node(base_retrieve_arxiv_full_text_node)
        base_extract_github_urls_node(base_extract_github_urls_node)
        base_summarize_paper_node(base_summarize_paper_node)
        base_select_best_paper_node(base_select_best_paper_node)
        generate_queries_node(generate_queries_node)
        add_web_scrape_node(add_web_scrape_node)
        add_extract_paper_title_node(add_extract_paper_title_node)
        add_search_arxiv_node(add_search_arxiv_node)
        add_retrieve_arxiv_full_text_node(add_retrieve_arxiv_full_text_node)
        add_extract_github_urls_node(add_extract_github_urls_node)
        add_summarize_paper_node(add_summarize_paper_node)
        add_select_best_paper_node(add_select_best_paper_node)
        prepare_state(prepare_state)
        __end__([<p>__end__</p>]):::last
        __start__ --> initialize_state;
        add_retrieve_arxiv_full_text_node --> add_extract_github_urls_node;
        add_search_arxiv_node --> add_retrieve_arxiv_full_text_node;
        add_web_scrape_node --> add_extract_paper_title_node;
        base_retrieve_arxiv_full_text_node --> base_extract_github_urls_node;
        base_search_arxiv_node --> base_retrieve_arxiv_full_text_node;
        base_select_best_paper_node --> generate_queries_node;
        base_web_scrape_node --> base_extract_paper_title_node;
        generate_queries_node --> add_web_scrape_node;
        initialize_state --> base_web_scrape_node;
        prepare_state --> __end__;
        base_extract_paper_title_node -. &nbsp;Stop&nbsp; .-> __end__;
        base_extract_paper_title_node -. &nbsp;Continue&nbsp; .-> base_search_arxiv_node;
        base_extract_github_urls_node -. &nbsp;Next paper&nbsp; .-> base_retrieve_arxiv_full_text_node;
        base_extract_github_urls_node -. &nbsp;Generate paper summary&nbsp; .-> base_summarize_paper_node;
        base_extract_github_urls_node -. &nbsp;All complete&nbsp; .-> base_select_best_paper_node;
        base_summarize_paper_node -. &nbsp;Next paper&nbsp; .-> base_retrieve_arxiv_full_text_node;
        base_summarize_paper_node -. &nbsp;All complete&nbsp; .-> base_select_best_paper_node;
        add_extract_paper_title_node -. &nbsp;Regenerate queries&nbsp; .-> generate_queries_node;
        add_extract_paper_title_node -. &nbsp;Continue&nbsp; .-> add_search_arxiv_node;
        add_extract_github_urls_node -. &nbsp;Next paper&nbsp; .-> add_retrieve_arxiv_full_text_node;
        add_extract_github_urls_node -. &nbsp;Generate paper summary&nbsp; .-> add_summarize_paper_node;
        add_extract_github_urls_node -. &nbsp;All complete&nbsp; .-> add_select_best_paper_node;
        add_summarize_paper_node -. &nbsp;Next paper&nbsp; .-> add_retrieve_arxiv_full_text_node;
        add_summarize_paper_node -. &nbsp;All complete&nbsp; .-> add_select_best_paper_node;
        add_select_best_paper_node -. &nbsp;Regenerate queries&nbsp; .-> generate_queries_node;
        add_select_best_paper_node -. &nbsp;Continue&nbsp; .-> prepare_state;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
```
</details>

## How to execute

```python
uv run python src/researchgraph/retrieve_paper_subgraph/retrieve_paper_subgraph.py
```
