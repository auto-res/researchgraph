import os
import pytest
import glob
from unittest.mock import patch
from test.writer_subgraph.utils.writer_subgraph_input_prep import WriterSubgraphInputPrep
from researchgraph.writer_subgraph.writer_subgraph import WriterSubgraph

SAMPLE_PAPER_DIR = "test/writer_subgraph/sample_papers/"
TEST_PAPERS = sorted(glob.glob(os.path.join(SAMPLE_PAPER_DIR, "*.pdf")))

OUTPUT_DIR = "/workspaces/researchgraph/test/writer_subgraph/output/"

llm_name = "gpt-4o-mini-2024-07-18"
data_prep = WriterSubgraphInputPrep(llm_name, OUTPUT_DIR)

@pytest.mark.parametrize("pdf_file", TEST_PAPERS)
def test_writer_subgraph_with_paper(pdf_file):
    writer_subgraph_input_data = data_prep.execute(pdf_file)
    print(f"writer_subgraph_input: {writer_subgraph_input_data}")

    latex_template_file_path = "/workspaces/researchgraph/data/latex/template.tex"
    figures_dir = "/workspaces/researchgraph/images"
    paper_name = os.path.splitext(os.path.basename(pdf_file))[0]
    pdf_file_path = os.path.join(OUTPUT_DIR, f"{paper_name}_generated.pdf")
    # llm_name = "gpt-4o-mini-2024-07-18"
    llm_name = "gpt-4o-2024-11-20"


    subgraph = WriterSubgraph(
        llm_name=llm_name,
        latex_template_file_path=latex_template_file_path,
        figures_dir=figures_dir,
        pdf_file_path=pdf_file_path, 
        refine_round = 2,
    ).build_graph()

    with patch("researchgraph.writer_subgraph.nodes.github_upload_node.GithubUploadNode.execute", return_value=True) as mock_github:
        result = subgraph.invoke(writer_subgraph_input_data)
        mock_github.assert_called_once()