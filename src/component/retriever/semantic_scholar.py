# %%
import os
import shutil
import requests
from langchain.document_loaders import PyPDFLoader
from semanticscholar import SemanticScholar

class SemanticScholarRetriever:
    def __init__(self, save_pdf_dir='data/PDF/'):
        self.save_pdf_dir = save_pdf_dir

    def download_from_arxiv_id(self, arxiv_id, save_dir):
        """Download PDF file from arXiv

        Args:
            arxiv_id (_type_): _description_
            save_dir (_type_): _description_
        """
        
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open(os.path.join(save_dir, f"{arxiv_id}.pdf"), 'wb') as file:
                shutil.copyfileobj(response.raw, file)
            print(f"Downloaded {arxiv_id}.pdf to {save_dir}")
        else:
            print(f"Failed to download {arxiv_id}.pdf")

    def download_from_arxiv_ids(self, arxiv_ids, save_dir):
        """Download PDF files from arXiv

        Args:
            arxiv_ids (_type_): _description_
            save_dir (_type_): _description_
        """
        # save_dirが存在しない場合、ディレクトリを作成
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else :
            shutil.rmtree(save_dir)
            os.makedirs(save_dir)

        for arxiv_id in arxiv_ids:
            self.download_from_arxiv_id(arxiv_id, save_dir)

    def convert_pdf_to_text(self, pdf_path):
        """Convert PDF file to text

        Args:
            pdf_path (_type_): _description_

        Returns:
            _type_: _description_
        """

        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        content = ""
        for page in pages[:20]:
            content += page.page_content

        return content

    def __call__(self, memory):
        """Retriever

        Args:
            memory (_type_): _description_
        """
        sch = SemanticScholar()
        results = sch.search_paper(memory["keywords"][0], limit=10)
        for item in results.items:
            print(item.title)
            print(item.paperId)

        DOI_ids = [item['externalIds'] for item in results.items]
        arxiv_ids = [item['ArXiv'] for item in DOI_ids if 'ArXiv' in item]

        self.download_from_arxiv_ids(arxiv_ids[:3], self.save_pdf_dir)

        # ディレクトリ内のすべてのPDFファイルを処理
        for idx, filename in enumerate(os.listdir(self.save_pdf_dir)):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.save_pdf_dir, filename)
                paper_content = self.convert_pdf_to_text(pdf_path)
                paper_key = f"paper_1_{idx+1}"
                memory["collection_of_papers_1"][paper_key] = paper_content
    
        return memory
    
    
if __name__ == "__main__":
    memory = {
        "keywords": ["llm", "optimizer", "loss function"],
        "collection_of_papers_1": {}
    }
    retriever = SemanticScholarRetriever()
    memory = retriever(memory)
    print(memory)