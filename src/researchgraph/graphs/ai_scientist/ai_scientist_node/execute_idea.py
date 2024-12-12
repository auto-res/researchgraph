from datetime import datetime
import os
import json
import shutil
import sys
import openai
from perform_experiments import ExperimentComponent
from perform_review import ReviewComponent, load_paper
from researchgraph.writingnode.writeup import WriteupComponent
from researchgraph.writingnode.draft_improvement import DraftImprovementComponent

from aider.io import InputOutput


def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class IdeaExecutionComponent:
    def __init__(self):
        pass

    def __call__(
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
        with open(os.path.join(base_dir, "run_0", "final_info.json"), "r") as f:
            baseline_results = json.load(f)
        baseline_results = {k: v["means"] for k, v in baseline_results.items()}
        exp_file = os.path.join(folder_name, "experiment.py")
        vis_file = os.path.join(folder_name, "plot.py")
        notes = os.path.join(folder_name, "notes.txt")
        with open(notes, "w") as f:
            f.write(f"# Title: {idea['Title']}\n")
            f.write(f"# Experiment description: {idea['Experiment']}\n")
            f.write("## Run 0: Baseline\n")
            f.write(f"Results: {baseline_results}\n")
            f.write("Description: Baseline results.\n")
        if log_file:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            log_path = os.path.join(folder_name, "log.txt")
            log = open(log_path, "a")
            sys.stdout = log
            sys.stderr = log
        io = InputOutput(
            yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt"
        )  # io は experiment でも　writeup でも共有なので、main.py で定義する
        try:
            print_time()
            print(f"*Starting idea: {idea_name}*")
            ## PERFORM EXPERIMENTS
            experiment_runner = ExperimentComponent(
                exp_file=exp_file,
                vis_file=vis_file,
                notes=notes,
                model=model,
                io=io,
            )
            print_time()
            print("*Starting Experiments*")
            try:
                memory_ = experiment_runner(
                    idea=idea,
                    memory_=memory_,
                    folder_name=folder_name,
                    baseline_results=baseline_results,
                )
                success = memory_["is_experiment_successful"]
                # success = perform_experiments(idea, folder_name, coder, baseline_results)
            except Exception as e:
                print(f"Error during experiments: {e}")
                print(f"Experiments failed for idea {idea_name}")
                memory_["is_idea_execution_successful"] = False
                return memory_

            if not success:
                print(f"Experiments failed for idea {idea_name}")
                memory_["is_idea_execution_successful"] = False
                return memory_

            print_time()
            print("*Starting Writeup*")
            ## PERFORM WRITEUP
            if writeup == "latex":
                writeup_file = os.path.join(folder_name, "latex", "template.tex")
                paper_writer = WriteupComponent(
                    exp_file=exp_file,
                    writeup_file=writeup_file,
                    notes=notes,
                    model=model,
                    io=io,
                )
                try:
                    memory_ = paper_writer(
                        idea, folder_name, client, client_model, memory_
                    )
                    # perform_writeup(idea, folder_name, coder, client, client_model)
                except Exception as e:
                    print(f"Failed to perform writeup: {e}")
                    memory_["is_idea_execution_successful"] = False
                    return memory_
                print("Done writeup")
            else:
                raise ValueError(f"Writeup format {writeup} not supported.")

            print_time()
            print("*Starting Review*")
            ## REVIEW PAPER
            if writeup == "latex":
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
                    # Store the review in separate review.txt file
                    with open(os.path.join(folder_name, "review.txt"), "w") as f:
                        f.write(json.dumps(review, indent=4))
                except Exception as e:
                    print(f"Failed to perform review: {e}")
                    memory_["is_idea_execution_successful"] = False
                    return memory_

            ## IMPROVE WRITEUP
            if writeup == "latex" and improvement:
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
                    draft_improver(review, folder_name, idea, memory_)

                    paper_text = load_paper(
                        f"{folder_name}/{idea['Name']}_improved.pdf"
                    )
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
                    # Store the review in separate review.txt file
                    with open(
                        os.path.join(folder_name, "review_improved.txt"), "w"
                    ) as f:
                        f.write(json.dumps(review))
                except Exception as e:
                    print(f"Failed to perform improvement: {e}")
                    memory_["is_idea_execution_successful"] = False
                    return memory_

            memory_["is_idea_execution_successful"] = True
            return memory_
        except Exception as e:
            print(f"Failed to evaluate idea {idea_name}: {str(e)}")
            memory_["is_idea_execution_successful"] = False
            return memory_
        finally:
            print("FINISHED IDEA")
            if log_file:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                log.close()
