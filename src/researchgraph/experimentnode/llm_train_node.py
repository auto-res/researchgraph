import os
import importlib.util
from typing import TypedDict
from langgraph.graph import StateGraph
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported


class State(TypedDict):
    script_save_path: str
    model_save_path: str


class LLMTrainNode:
    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        model_name: str,
        dataset_name: str,
        model_save_path: str,
        num_train_data: int | None = None,
    ):
        self.input_variable = input_variable
        self.output_variable = output_variable
        print("LLMTrainNode")
        print(f"input: {self.input_variable}")
        print(f"output: {self.output_variable}")
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.model_save_path = model_save_path
        self.num_train_data = num_train_data
        self.training_args = self._set_up_training_args()
        self.model, self.tokenizer = self._set_up_model()
        self.dataset = self._set_up_dataset()

    def _set_up_training_args(self):
        training_args_kwargs = {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 5,
            "learning_rate": 2e-4,
            "fp16": not is_bfloat16_supported(),
            "bf16": is_bfloat16_supported(),
            "logging_steps": 1,
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "seed": 3407,
            "output_dir": "outputs",
            "report_to": "none",
        }

        if self.num_train_data is None:
            training_args_kwargs["num_train_epochs"] = 1
        else:
            training_args_kwargs["max_steps"] = self.num_train_data

        training_args = TrainingArguments(**training_args_kwargs)
        return training_args

    def _set_up_model(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        return model, tokenizer

    def _set_up_dataset(self):
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

    def __call__(self, state: State) -> dict:
        script_path = state[self.input_variable]
        new_optimizer = self._set_up_optimizer(script_path)
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            dataset_num_proc=2,
            packing=False,
            optimizers=(new_optimizer, None),
            args=self.training_args,
        )
        trainer_stats = trainer.train()

        self.model.save_pretrained(self.model_save_path)
        self.tokenizer.save_pretrained(self.model_save_path)
        return {self.output_variable: self.model_save_path}


if __name__ == "__main__":
    model_name = "unsloth/Meta-Llama-3.1-8B"
    dataset_name = "openai/gsm8k"
    model_save_path = "model"
    input_variable = ""
    output_variable = "model_save_path"

    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "llmtrainer",
        LLMTrainNode(
            model_name=model_name,
            dataset_name=dataset_name,
            model_save_path=model_save_path,
            input_variable="script_save_path",
            output_variable="model_save_path",
        ),
    )
    graph_builder.set_entry_point("llmtrainer")
    graph_builder.set_finish_point("llmtrainer")
    graph = graph_builder.compile()

    memory = {
        "script_save_path": "/content/new_method.py",
    }

    graph.invoke(memory)
