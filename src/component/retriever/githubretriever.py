import subprocess
import glob

def get_folder_structure(path='.'):
    try:
        # subprocess.runを使用してフォルダ構造を取得
        result = subprocess.run(['ls', '-R', path], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return None

def git_clone(url):
    command = f'git clone {url}'
    ls = subprocess.run(command, shell=True, text=True)
    return ls

def get_py_files(url):
    folder_structure = get_folder_structure(url.split('/')[-1])
    py_filelist = glob.glob(url.split('/')[-1]+"/**/*.py", recursive=True)
    py_text = []
    for i in py_filelist:
      with open(i) as f:
          file_read = f.read()
      py_text.append('<FILE='+i+'> \n'+file_read.replace('\n\n', '\n').replace('\n\n', '\n')+'</FILE> \n')
      result = ''.join(py_text)
    return folder_structure,result



def GithubRetriever(input):
    github_url = input["github_url"]
    ls = git_clone(github_url)
    folder_structure,get_file = get_py_files(github_url)
    output = {
    "folder_structure" : folder_structure,
    "github_file" : get_file
    }
    return output


GithubRetriever_output = GithubRetriever(input)
