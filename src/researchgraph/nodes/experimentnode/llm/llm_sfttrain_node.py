import os
import io
import sys
import importlib.util
import traceback
#from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

from researchgraph.core.node import Node

from typing import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph


class State(BaseModel):
    script_save_path: str = Field(default="")
    model_save_path: str = Field(default="") 
    log_save_path: str = Field(default="")    
    error_log_save_path: str = Field(default="")
    execution_flag_list: list = Field(default_factory=list)


class LLMSFTTrainNode(Node):
    def __init__(
        self,
        input_key: list[str],
        output_key: list[str],
        model_name: str,
        dataset_name: str,
        model_save_path: str,
        lora: bool = False,
        num_train_data: int | None = None,
    ):
        super().__init__(input_key, output_key)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.model_save_path = model_save_path
        self.lora = lora
        self.num_train_data = num_train_data
        self.sft_args = self._set_up_training_args()
        self.model, self.tokenizer = self._set_up_model()
        self.dataset = self._set_up_dataset()

    def _set_up_training_args(self):
        sft_args = SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            learning_rate=2e-4,
            logging_steps=1,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
            dataset_num_proc=2,
            packing=False,
            dataset_text_field="text",
            max_seq_length=2048,  
            #max_steps=self.num_train_data,
            logging_first_step=True,
        )

        if self.num_train_data is None:
            sft_args.num_train_epochs = 1
        else:
            sft_args.max_steps = self.num_train_data
        return sft_args

    def _set_up_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token_id = model.config.eos_token_id
        return model, tokenizer

    def _set_up_dataset(self):
        print(f"Dataset name: {self.dataset_name}")
        dataset = load_dataset(self.dataset_name, "main")
        dataset = dataset["train"].map(self._formatting_prompts_func, batched=True)
        return dataset

    def _formatting_prompts_func(self, examples):
        prompt = """### Input:
        {input}
        ### Output:
        {output}"""
        inputs = examples["question"]
        outputs = examples["answer"]
        texts = []
        EOS_TOKEN = self.tokenizer.eos_token
        for input, output in zip(inputs, outputs):
            text = prompt.format(input=input, output=output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    def _set_up_optimizer(self, script_path: str):
        module_name = os.path.splitext(os.path.basename(script_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        NewOptimizer = module.NewOptimizer
        new_optimizer = NewOptimizer(self.model.parameters())
        return new_optimizer
    
    def redirect_stdout_to_string(self):
        """
        Redirects standard output to a string buffer.
        Returns the buffer and the original stdout.
        """
        buffer = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = buffer
        return buffer, original_stdout

    def redirect_stderr_to_string(self):
        """
        Redirects standard error to a string buffer.
        Returns the buffer and the original stderr.
        """
        buffer = io.StringIO()
        original_stderr = sys.stderr
        sys.stderr = buffer
        return buffer, original_stderr

    def restore_streams(self, buffer, original_stream):
        """
        Restores the original stdout or stderr and retrieves buffer content.
        """
        sys.stdout = original_stream if original_stream is not None else sys.stdout
        sys.stderr = original_stream if original_stream is not None else sys.stderr
        return buffer.getvalue()

    def execute(self, state) -> dict:
        script_path = getattr(state, self.input_key[0])
        execution_flag_list = getattr(state, self.output_key[3])

        log_output = ""
        error_log_output = ""
    
        # NOTE: 
        try:
            stdout_buffer, original_stdout = self.redirect_stdout_to_string()

            new_optimizer = self._set_up_optimizer(script_path)
            # NOTE: Please refer to the following link for SFTTrainer
            # https://huggingface.co/docs/trl/sft_trainer
            trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.dataset,
                optimizers=(new_optimizer, None),
                args=self.sft_args,
            )
            trainer.train()
            execution_flag_list.append(True)
        except Exception as e:
            execution_flag_list.append(False)
            # NOTE:
            stderr_buffer, original_stderr = self.redirect_stderr_to_string()
            try:
                print("An error has occurred.", file=sys.stderr)
                print(f"Error type: {type(e).__name__}", file=sys.stderr)
                print(f"Error message: {e}", file=sys.stderr)
                print("Traceback:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
            finally:
                error_log_output = self.restore_streams(stderr_buffer, original_stderr)
        finally:
            log_output = self.restore_streams(stdout_buffer, original_stdout)
    
        self.model.save_pretrained(self.model_save_path)
        self.tokenizer.save_pretrained(self.model_save_path)
        return {
            self.output_key[0]: log_output,
            self.output_key[1]: self.model_save_path,
            self.output_key[2]: error_log_output,
            self.output_key[3]: execution_flag_list,
            }


# NOTE: We have not implemented the test code in pytest because we do not have a GPU environment for testing with GitHub Actions.
if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-3B"
    dataset_name = "openai/gsm8k"
    model_save_path = "/workspaces/researchgraph/data/model"
    input_key = ["script_save_path"]
    output_key = ["logs","model_save_path", "error_logs", "execution_flag_list"]
    num_train_data = 5

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "llmsfttrainer",
        LLMSFTTrainNode(
            input_key=input_key,
            output_key=output_key,
            model_name=model_name,
            dataset_name=dataset_name,
            model_save_path=model_save_path,
            num_train_data=num_train_data,
        ),
    )
    graph_builder.set_entry_point("llmsfttrainer")
    graph_builder.set_finish_point("llmsfttrainer")
    graph = graph_builder.compile()

    state = {
        "script_save_path": "/workspaces/researchgraph/test/experimentnode/new_method.py",
    }

    graph.invoke(state, debug=True)
