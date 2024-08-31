from llm_component.llm_component import LLMComponent
from component.retriever.semantic_scholar import SemanticScholarRetriever

llm_name = 'gemini-1.5-pro'



retriever = SemanticScholarRetriever()
keyworder = LLMComponent(
    llm_name=llm_name,
    json_file_path='./llm_component/keyworder/keyworder.json'
    )
selector = LLMComponent(
    llm_name=llm_name,
    json_file_path='./llm_component/retriever/retriever.json'
    )



memory = {
    
}
