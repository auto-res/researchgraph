from datetime import datetime
import os
import json
import shutil
import sys
import openai
from researchgraph.graph.ai_scientist.ai_scientist_node.perform_experiments import ExperimentComponent
from researchgraph.graph.ai_scientist.ai_scientist_node.perform_writeup import WriteupComponent, DraftImprovementComponent
from researchgraph.graph.ai_scientist.ai_scientist_node.perform_review import ReviewComponent, load_paper
from researchgraph.writingnode.texnode import TextNode

from aider.io import InputOutput


def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class IdeaExecutionComponent:
    def __init__(self):
        pass

    def __call__(
        self, 
        base_dir,
        results_dir,
        idea,
        model,
        client,
        client_model,
        writeup,
        improvement,
        memory_,
        log_file=False,
    ):
        ## CREATE PROJECT FOLDER
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        idea_name = f"{timestamp}_{idea['Name']}"
        folder_name = os.path.join(results_dir, idea_name)
        assert not os.path.exists(folder_name), f"Folder {folder_name} already exists."
        destination_dir = folder_name
        shutil.copytree(base_dir, destination_dir, dirs_exist_ok=True)

        baseline_results = self.load_baseline_results(base_dir)
        exp_file, vis_file, notes = self.setup_experiment_files(folder_name, idea, baseline_results)
        log, original_stdout, original_stderr = self.setup_logging(folder_name, log_file)

        io = InputOutput(
            yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt"
        )  # io は experiment でも　writeup でも共有なので、main.py で定義する

        try:
            print_time()
            print(f"*Starting idea: {idea_name}*")

            # PERFORM EXPERIMENTS
            if not self.perform_experiments(exp_file, vis_file, notes, model, io, idea, memory_, folder_name, baseline_results):
                return memory_
            
            # PERFORM WRITEUP
            if writeup == "latex":
                writeup_file = os.path.join(folder_name, "latex", "template.tex")
                if not self.perform_writeup(writeup_file, exp_file, notes, model, io, idea, folder_name, client, client_model, memory_):
                    return memory_
            else:
                raise ValueError(f"Writeup format {writeup} not supported.")
            
            # Generate LaTeX PDF
            if not self.generate_pdf(folder_name, idea, memory_):
                return memory_

            # Review Paper
            if writeup == "latex":
                review = self.perform_review(folder_name, idea, memory_)
                if review is None:
                    return memory_
                memory_["review"] = review
                
            # Improve Writeup
            if writeup == "latex" and improvement:
                if not self.improve_writeup(writeup_file, exp_file, notes, model, io, review, memory_, folder_name, idea):
                    return memory_

            memory_["is_idea_execution_successful"] = True
            return memory_
        except Exception as e:
            print(f"Failed to evaluate idea {idea_name}: {str(e)}")
            memory_["is_idea_execution_successful"] = False
            return memory_
        finally:
            print("FINISHED IDEA")
            self.teardown_logging(log_file, log, original_stdout, original_stderr)

    def load_baseline_results(self, base_dir):
        with open(os.path.join(base_dir, "run_0", "final_info.json"), "r") as f:
            baseline_results = json.load(f)
        return {k: v["means"] for k, v in baseline_results.items()}
    
    def setup_experiment_files(self, folder_name, idea, baseline_results):
        exp_file = os.path.join(folder_name, "experiment.py")
        vis_file = os.path.join(folder_name, "plot.py")
        notes = os.path.join(folder_name, "notes.txt")
        with open(notes, "w") as f:
            f.write(f"# Title: {idea['Title']}\n")
            f.write(f"# Experiment description: {idea['Experiment']}\n")
            f.write("## Run 0: Baseline\n")
            f.write(f"Results: {baseline_results}\n")
            f.write("Description: Baseline results.\n")
        return exp_file, vis_file, notes
    
    def setup_logging(self, folder_name, log_file):
        if log_file:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            log_path = os.path.join(folder_name, "log.txt")
            log = open(log_path, "a")
            sys.stdout = log
            sys.stderr = log
            return log, original_stdout, original_stderr
        return None, None, None
    
    def teardown_logging(self, log_file, log, original_stdout, original_stderr):
        if log_file:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log.close()
    
    def perform_experiments(
            self, 
            exp_file, 
            vis_file, 
            notes, 
            model, 
            io, 
            idea, 
            memory_, 
            folder_name, 
            baseline_results
    ):
            print_time()
            print("*Starting Experiments*")
            experiment_runner = ExperimentComponent(
                exp_file=exp_file,
                vis_file=vis_file,
                notes=notes,
                model=model,
                io=io,
            )
            try:
                memory_ = experiment_runner(
                    idea=idea,
                    memory_=memory_,
                    folder_name=folder_name,
                    baseline_results=baseline_results,
                )
                return memory_["is_experiment_successful"]
            except Exception as e:
                print(f"Error during experiments: {e}")
                memory_["is_idea_execution_successful"] = False
                return False
            
    def perform_writeup(
       self, 
       writeup_file, 
       exp_file, 
       notes, 
       model, 
       io, 
       idea, 
       folder_name, 
       client, 
       client_model, 
       memory_     
    ):
            print_time()
            print("*Starting Writeup*")
            paper_writer = WriteupComponent(
                exp_file=exp_file,
                writeup_file=writeup_file,
                notes=notes,
                model=model,
                io=io,
            )
            try:
                memory_ = paper_writer(idea, folder_name, client, client_model, memory_)
                print("Done writeup")
                return True
            except Exception as e:
                print(f"Failed to perform writeup: {e}")
                memory_["is_idea_execution_successful"] = False
                return False
 

    def generate_pdf(self, folder_name, idea, memory_):
        print_time()
        print("*Generating LaTeX PDF*")
        try:
            coder_out = memory_["writeup_content"]
            text_node = TextNode(coder_out)
            text_node.setup_latex_utils()
            text_node.generate_latex(
                folder_name=folder_name,
                pdf_file=os.path.join(folder_name, f"{idea['Name']}.pdf")
            )
            return True
        except Exception as e:
            print(f"Failed to generate LaTeX PDF: {e}")
            memory_["is_idea_execution_successful"] = False
            return False

    def perform_review(self, folder_name, idea, memory_):
        print_time()
        print("*Starting Review*")
        try:
            paper_text = load_paper(f"{folder_name}/{idea['Name']}.pdf")
            reviewer = ReviewComponent()
            memory_ = reviewer(
                paper_text,
                model="gpt-4o-2024-05-13",
                client=openai.OpenAI(),
                memory_=memory_,
                num_reflections=5,
                num_fs_examples=1,
                num_reviews_ensemble=5,
                temperature=0.1,
            )
            review = memory_["review"]
            with open(os.path.join(folder_name, "review.txt"), "w") as f:
                f.write(json.dumps(review, indent=4))
            return True
        except Exception as e:
            print(f"Failed to perform review: {e}")
            memory_["is_idea_execution_successful"] = False
            return False

    def improve_writeup(self, writeup_file, exp_file, notes, model, io, review, memory_, folder_name, idea):
        print_time()
        print("*Starting Improvement*")
        try:
            draft_improver = DraftImprovementComponent(
                writeup_file=writeup_file,
                exp_file=exp_file,
                notes=notes,
                model=model,
                io=io,
            )
            draft_improver(review, memory_)
            return self.generate_pdf(folder_name, idea, memory_)
        except Exception as e:
            print(f"Failed to perform improvement: {e}")
            memory_["is_idea_execution_successful"] = False
            return False