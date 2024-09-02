import os
import subprocess
import glob

class GithubRetriever:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        print("GithubRetriever initialized")
        print("input: ['github_url']")
        print("output: ['folder_structure', 'github_file']")
    
    def get_folder_structure(self, path='.'):
        try:
            # subprocess.runを使用してフォルダ構造を取得
            result = subprocess.run(['ls', '-R', path], capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
            return None

    def git_clone(self, url):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        os.chdir(self.save_dir)
        command = f'git clone {url}'
        ls = subprocess.run(command, shell=True, text=True)
        return ls

    def get_py_files(self, url):
        folder_structure = self.get_folder_structure(url.split('/')[-1])
        py_filelist = glob.glob(url.split('/')[-1]+"/**/*.py", recursive=True)
        py_text = []
        for i in py_filelist:
            with open(i) as f:
                file_read = f.read()
        py_text.append('<FILE='+i+'> \n'+file_read.replace('\n\n', '\n').replace('\n\n', '\n')+'</FILE> \n')
        result = ''.join(py_text)
        return folder_structure,result

    def __call__(self, memory):
        github_url = memory["github_url"]
        self.git_clone(github_url)
        folder_structure,get_file = self.get_py_files(github_url)
        memory["folder_structure"] = folder_structure
        memory["github_file"] = get_file
        return memory


if __name__ == "__main__":
    save_dir = "/workspaces/researchchain/data"
    memory = {
        "github_url": "https://github.com/fuyu-quant/IBLM"
    }
    githubretriever = GithubRetriever(save_dir=save_dir)
    memory = githubretriever(memory)
    print(memory)
