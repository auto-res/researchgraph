import os

GITHUB_WORKSPACE = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
TEST_PDF_FILE = os.path.join(GITHUB_WORKSPACE, "data/test_output.pdf")

writer_subgraph_input_data = {
    "objective": "Researching optimizers for fine-tuning LLMs.",
    "base_method_text": "Baseline method description...",
    "add_method_text": "Added method description...",
    "new_method_text": ["New combined method description..."],
    "base_method_code": "def base_method(): pass",
    "add_method_code": "def add_method(): pass",
    "new_method_code": ["def new_method(): pass"],
    "base_method_results": "Accuracy: 0.85",
    "add_method_results": "Accuracy: 0.88",
    "new_method_results": ["Accuracy: 0.92"],
    "arxiv_url": "https://arxiv.org/abs/1234.5678",
    "github_url": "https://github.com/example/repo",
    "paper_content": {}, 
    "pdf_file_path": TEST_PDF_FILE,  
}
