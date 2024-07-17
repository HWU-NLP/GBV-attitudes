"""
https://www.mlexpert.io/machine-learning/tutorials/alpaca-fine-tuning#inference
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
https://docs.wandb.ai/guides/track/environment-variables
"""
import os
import argparse
import json
from pathlib import Path
from typing import List
import wandb

import torch
from datasets import Dataset, Features, Value, Sequence
from tokenizer import CustomLlamaTokenizer
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
from transformers import TrainingArguments, DataCollatorForSeq2Seq, Trainer
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)
from huggingface_hub import login
# hub_token = "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"
hub_token = "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"
login(token=hub_token)


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
print(torch.cuda.device_count())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAMES = {
                  "llama": "decapoda-research/llama-7b-hf",
                  "llama2-chat": "meta-llama/Llama-2-7b-chat-hf",
                  "llama2": "meta-llama/Llama-2-7b-hf",
                  "llama2-13b": "meta-llama/Llama-2-13b-hf",
                  "llama2-70b": "meta-llama/Llama-2-70b-hf",
                  "llama3": "meta-llama/Meta-Llama-3-8B",
              } 

schema = Features({
    'instruction': Value(dtype='string'),
    'input': Value(dtype='string'),
    'choices': Sequence(Value(dtype='string')),  
    'output': Value(dtype='string'),
    'output_anno': Value(dtype='string'),
    'anno_id': Value(dtype='string'),
    # 'anno_info': Value(dtype='dict'),  
    'anno_prompt': Value(dtype='string'),
    'anno_short_prompt': Value(dtype='string'),
    'id': Value(dtype='string'),
    'prompt': Value(dtype='string'),
    'input_ids': Sequence(Value(dtype='int32')),  
    'attention_mask': Sequence(Value(dtype='int32')),  
    'labels': Sequence(Value(dtype='int32')), 
    'do_truncate': Value(dtype='bool'),
    'truncate_mode': Value(dtype='string'),
    'n_truncated_tokens': Value(dtype='int32')
})


def get_dirs(alpaca_format, dataset, feature_name, model_name, answer_type, use_demo):

    feature_root_dir = f"feature-{feature_name}" 
    model_root_dir = f"model/{model_name}-{feature_name}"
    name = ( 
        f"{dataset}-{alpaca_format}-{answer_type}-{model_name}-demo" if use_demo 
        else f"{dataset}-{alpaca_format}-{answer_type}-{model_name}"
        )

    wandb_run_name = f"gbv_ft600_{feature_name}-{name}"
    train_dataset_dir = Path(feature_root_dir, name, "train.json")
    val_dataset_dir = Path(feature_root_dir, name, "val.json")
    output_dir = Path(model_root_dir, name)

    return wandb_run_name, train_dataset_dir, val_dataset_dir, output_dir


def load_dataset(data_dir):
    # data = Dataset.from_list(torch.load(data_dir))
    with open(data_dir, "r") as file:
        loaded_data = json.load(file)
    for d in loaded_data:
        del d['anno_info']
    data = Dataset.from_list(loaded_data, features=schema)

    return data


def loss_fn(logits, labels):
    # shift the labels such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()

    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)

    loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)# , ignore_index=-1 if mask_input=True else default -100
    return loss


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, inputs.get("labels"))

        return (loss, outputs) if return_outputs else loss


def main(
    # data/model params
    dataset: str,
    alpaca_format: str,
    model_name: str,
    use_demo: bool,
    answer_type: str,
    feature_name: str,
    # training hyperparams
    num_epochs: int = 3, # 40 for samples / default = 3 for full dataset
    batch_size: int = 128,
    eval_save_steps: int = 100, # 40 for samples / 100 for full dataset
    micro_batch_size: int = 4,  
    learning_rate: float = 3e-4,
    weight_decay: float = 0.001,
    optim: str = "adamw_torch",
    fp16: bool = True,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj","v_proj"],
    # wandb params
    wandb_watch: str = "false",  # options: false | gradients | all
    wandb_log_model: str = "true",  # options: false | true
    wandb_api_key: str = "92c2a83f3839d4853f524a71f35a176f84c2697d",
    ) -> None:
    """Process original datasets into alpaca-like format and save into json files.
    """

    # Check if parameter passed or if set within environ
    # Only overwrite environ if wandb param passed
    wandb_project = "gbv_model" 
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_WATCH"] = wandb_watch
    os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    os.environ["WANDB_API_KEY"] = wandb_api_key

    wandb_run_name, train_dataset_dir, val_dataset_dir, output_dir = (
        get_dirs(alpaca_format, dataset, feature_name, model_name, answer_type, use_demo)
        )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n#### Loading {dataset}'s train data in {alpaca_format} format from {str(train_dataset_dir)}")
    print(f"#### Loading {dataset}'s validation data in {alpaca_format} format from {str(val_dataset_dir)}")
    print(f"set use_demo={str(use_demo)}")
    train_data = load_dataset(train_dataset_dir)
    val_data = load_dataset(val_dataset_dir)

    print(f"\n#### Loading model - {MODEL_NAMES[model_name]}")
    # tokenizer = CustomLlamaTokenizer.from_pretrained(MODEL_NAMES[model_name]) - for llama2
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[model_name])
    tokenizer.add_special_tokens({"pad_token":"<pad>"}) 
    tokenizer.pad_token_id = 0  # unk
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"
    model = LlamaForCausalLM.from_pretrained(MODEL_NAMES[model_name],
                                            # params for big model inference
                                            torch_dtype=torch.float16,
                                            device_map='auto', # A map that specifies where each submodule should go.
                                            load_in_8bit=True, # convert the loaded model into mixed-8bit quantized model (need to install bitsandbytes)
                                            # llm_int8_enable_fp32_cpu_offload=True
                                            )

    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.config.pad_token_id = 0  
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.print_trainable_parameters()
    model.to(DEVICE)

    torch.cuda.empty_cache()

    print("\n#### Training model ...")
    training_args = TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=batch_size//micro_batch_size,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        # max_steps=max_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        logging_strategy="epoch",
        logging_steps=10,
        optim=optim,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_save_steps, 
        save_steps=eval_save_steps, 
        output_dir=output_dir,
        load_best_model_at_end=True,
        report_to=["wandb"],
        run_name=wandb_run_name,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    trainer = MyTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=data_collator,
    )

    # model = torch.compile(model)
    trainer.train()

    print(f"\n#### Saving fine-tuned model -- {output_dir}")
    model.save_pretrained(output_dir)

    print("\n#### Evaluating model ...")
    trainer.evaluate()
    # Validate the model
    val_results = trainer.evaluate(eval_dataset=val_data)
    print(val_results)

    # Save evaluation results
    with open(Path(output_dir, "val_results.txt"), "w") as f:
        for metric, value in val_results.items():
            f.write(f"{metric}: {value}\n")

    wandb.finish()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='mix', help="gbv dataset")
    parser.add_argument("--alpaca_format", type=str, default='alpaca-option2', choices=["alpaca-option2"], help="alpaca instruction format")
    parser.add_argument("--model_name", type=str, default='llama2', help="pre-trained model used")
    parser.add_argument("--use_demo", action="store_true", help="whether or not use prompt with demonstrations")
    parser.add_argument("--answer_type", type=str, default='harness', choices=["harness","helm"], help="evaluation format")
    parser.add_argument("--num_epochs", type=int, default=5, help=" total number of training epochs, set 100 for samples")
    parser.add_argument("--eval_save_steps", type=int, default=100, help=" total number of eval/save steps, set 50 for samples")
    parser.add_argument("--feature_name", type=str, default='first', help="processed feature name")
    
    args = parser.parse_args()

    main(dataset=args.dataset, 
        alpaca_format=args.alpaca_format, 
        model_name=args.model_name, 
        use_demo=args.use_demo,
        answer_type=args.answer_type,
        num_epochs=args.num_epochs,
        eval_save_steps=args.eval_save_steps,
        feature_name=args.feature_name,
        )