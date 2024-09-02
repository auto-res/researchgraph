# %%
from llm_component.llm_component import LLMComponent
from component.retriever.semantic_scholar import SemanticScholarRetriever
from component.retriever.github import GithubRetriever

if __name__ == "__main__":
    llm_name = 'gemini-1.5-pro'
    save_dir = "/workspaces/researchchain/data"

    retriever = SemanticScholarRetriever(save_dir=save_dir)
    keyworder = LLMComponent(
        llm_name=llm_name,
        json_file_path='llm_component/hotl/keyworder/keyworder.json'
        )
    selector = LLMComponent(
        llm_name=llm_name,
        json_file_path='./llm_component/hotl/selector/selector1.json'
        )
    extractor = LLMComponent(
        llm_name=llm_name,
        json_file_path='./llm_component/hotl/extractor/extractor_pre_method.json'
        )
    githubretriever = GithubRetriever(save_dir=save_dir)
    codeextractor = LLMComponent(
        llm_name=llm_name,
        json_file_path='./llm_component/hotl/codeextractor/codeextractor_patch_method.json'
        )
    decomposer = LLMComponent(
        llm_name=llm_name,
        json_file_path='./llm_component/hotl/decomposer/decomposer_patch_method.json'
        )

    memory = {
        'environment' : 'The following two experimental environments are available・Fine tuning of the LLM and experiments with rewriting the Optimizer or loss function.・Verification of the accuracy of prompt engineering.',
        'objective' : 'Combining two papers to create a new methodology',
        'keywords' : 'LLM'
    }

    #memory = keyworder(memory)
    #print(memory)
    memory = retriever(memory)
    print(memory)
    #memory = selector(memory)
    #memory = extractor(memory)
    #memory = githubretriever(memory)
    #memory = codeextractor(memory)
    #memory = decomposer(memory)

