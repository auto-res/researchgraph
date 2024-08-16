import os
import shutil
import requests
import json
import glob
from tqdm import tqdm
from langchain.document_loaders import PyPDFLoader
from semanticscholar import SemanticScholar

def download_from_arxiv_id(arxiv_id, save_dir):
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    # PDFファイルをダウンロード
    response = requests.get(url, stream=True)

    # ダウンロードが成功したかチェック
    if response.status_code == 200:
        # ファイルに書き込む
        with open(os.path.join(save_dir, f"{arxiv_id}.pdf"), 'wb') as file:
            shutil.copyfileobj(response.raw, file)
        print(f"Downloaded {arxiv_id}.pdf to {save_dir}")
    else:
        print(f"Failed to download {arxiv_id}.pdf")

def download_from_arxiv_ids(arxiv_ids, save_dir):
    # save_dirが存在しない場合、ディレクトリを作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else :
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    for arxiv_id in arxiv_ids:
        download_from_arxiv_id(arxiv_id, save_dir)

def convert_pdf_to_text(pdf_path):

    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    content = ""
    for page in pages[:20]:
        content += page.page_content

    return content

def retriever(file_path = 'data/keyworder_output.json'):

    # PDFファイルが保存されているディレクトリ
    pdf_directory = 'data/PDF/'

    # JSONファイルを読み込みます
    with open(file_path, 'r') as file:
        retreiver_input = json.load(file)

    # 読み込んだ内容を確認します
    print(retreiver_input["keywords"][0])

    sch = SemanticScholar()
    results = sch.search_paper(retreiver_input["keywords"][0], limit=10)
    for item in results.items:
        print(item.title)
        print(item.paperId)

    DOI_ids = [item['externalIds'] for item in results.items]
    arxiv_ids = [item['ArXiv'] for item in DOI_ids if 'ArXiv' in item]

    download_from_arxiv_ids(arxiv_ids[:3], pdf_directory)

    # 出力用の辞書
    output_data = {"collection_of_papers_1": {}}

    # ディレクトリ内のすべてのPDFファイルを処理
    for idx, filename in enumerate(os.listdir(pdf_directory)):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            paper_content = convert_pdf_to_text(pdf_path)
            paper_key = f"paper_1_{idx+1}"
            output_data["collection_of_papers_1"][paper_key] = paper_content

    # JSONファイルに保存
    with open('data/Retriever_output.json', 'w', encoding='utf-8') as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=4)
