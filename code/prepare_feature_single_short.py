"""
https://github.com/sinantie/lit-llama/blob/593b6ea562f5da514ad362bb414afbe6b9cc4bd0/scripts/prepare_alpaca.py
"""
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean, mode, median

from tokenizer import CustomLlamaTokenizer
from transformers import AutoTokenizer, T5Tokenizer, LlamaTokenizer
from huggingface_hub import login
hub_token = "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"
login(token=hub_token)

INSTRUCTION = "Classify the following message from a social media platform. It might contain a form of gender-based violence (GBV). Output A if it contains GBV, or B if not."
CHOICES = ["A. GBV", "B. Not GBV"]

MODEL_NAMES = {
                  "llama": "decapoda-research/llama-7b-hf",
                  "llama2": "meta-llama/Llama-2-7b-hf",
                  "llama2-13b": "meta-llama/Llama-2-13b-hf",
                  "llama2-70b": "meta-llama/Llama-2-70b-hf",
                  "llama3": "meta-llama/Meta-Llama-3-8B",
                  "flant5": "google/flan-t5-large",
                  "flant5-xl": "google/flan-t5-xl",
                  "flant5-small": "google/flan-t5-small",
                  "mistral-v3": "mistralai/Mistral-7B-v0.3",
                  "mistral": "mistralai/Mistral-7B-v0.1",
              } 

IGNORE_INDEX = -100


def generate_demo(model_name: str, answer_type: str, use_demo: bool):
    """ 
    generate demonstrations for prompt
    """
    choice = "Choices: A. GBV or B. Not GBV"
    text1 = "Fuck the niggers and the jews. Both have a history of eating white people and fuck you for suggesting I should get along with people who want to eat me."
    anno11 = "This annotator is a 32-year-old white female, who is bisexual and left-wing/liberal politics.\nThree scales are used to show the annotator's attitudes, namely the Very Short Authoritarianism (VSA) scale to measure Right Wing Authoritarianism (RWA), the Short Social Dominance Orientation (SSDO) scale to measure Social Dominance Orientation (SDO), and the Brief Hostile Neosexism scale to measure Hostile Neosexism. This worker is moderate right wing authoritarianism, high social dominance orientation, and high hostile neosexism."
    label11 = "A. GBV"
    anno12 = "This annotator is a 48-year-old asian female, who is heterosexual and right-wing/conservative politics.\nThree scales are used to show the annotator's attitudes, namely the Very Short Authoritarianism (VSA) scale to measure Right Wing Authoritarianism (RWA), the Short Social Dominance Orientation (SSDO) scale to measure Social Dominance Orientation (SDO), and the Brief Hostile Neosexism scale to measure Hostile Neosexism. This worker is moderate right wing authoritarianism, moderate social dominance orientation, and high hostile neosexism."
    label12 = "B. Not GBV"
    
    text2 = "Well then good because someone has to knock her ass back in line when she acts like a woman and posts bullshit about her (((based minorities))). If someone is not white they are NOT one of us."
    anno21 = "This annotator is a 35-year-old american male, who is bisexual and right-wing/conservative politics.\nThree scales are used to show the annotator's attitudes, namely the Very Short Authoritarianism (VSA) scale to measure Right Wing Authoritarianism (RWA), the Short Social Dominance Orientation (SSDO) scale to measure Social Dominance Orientation (SDO), and the Brief Hostile Neosexism scale to measure Hostile Neosexism. This worker is moderate right wing authoritarianism, moderate social dominance orientation, and high hostile neosexism."
    label21 = "A. GBV"
    anno22 = "This annotator is a 32-year-old asian female, who is heterosexual and right-wing/conservative politics.\nThree scales are used to show the annotator's attitudes, namely the Very Short Authoritarianism (VSA) scale to measure Right Wing Authoritarianism (RWA), the Short Social Dominance Orientation (SSDO) scale to measure Social Dominance Orientation (SDO), and the Brief Hostile Neosexism scale to measure Hostile Neosexism. This worker is moderate right wing authoritarianism, moderate social dominance orientation, and moderate hostile neosexism."
    label22 = "B. Not GBV"
    
    # default answer format is "helm"     
    demo11 = f"{anno11} Text: {text1} {choice} Answer: {label11[0]}"
    demo12 = f"{anno12} Text: {text1} {choice} Answer: {label12[0]}"
    demo21 = f"{anno21} Text: {text2} {choice} Answer: {label21[0]}"
    demo22 = f"{anno22} Text: {text2} {choice} Answer: {label22[0]}"

    # specify demo format for answer format "harness"
    if answer_type == "harness":
        demo11 = f"{demo11[:-1]}{label11}"
        demo12 = f"{demo12[:-1]}{label12}"
        demo21 = f"{demo21[:-1]}{label21}"
        demo22 = f"{demo22[:-1]}{label22}"

    demo = f"{demo11} {demo12} {demo21} {demo22}"
    return  demo if use_demo else ""


def generate_prompt(sample: dict, model_name: str, answer_type: str, use_demo: bool, label_name: str):
    """ 
    Generate prompt according to different model format needs
    Add demonstrations between instruction if required
    """
    demo = generate_demo(model_name, answer_type, use_demo)
    input = sample["input"]
    anno_prompt = sample["anno_short_prompt"]
    choice = f"Choices: A. GBV or B. Not GBV"
    
    if label_name == "single-x":
        # format prompt with demonstration
        prompt_demo = f"\{INSTRUCTION} {demo} Text: {input} {choice} Answer:" # without anno info
        # format prompt without demonstration
        prompt = f"\{INSTRUCTION} Text: {input} {choice} Answer:" # without anno info

    elif label_name == "single-short":
        # format prompt with demonstration
        prompt_demo = f"\{INSTRUCTION} {demo} {anno_prompt} Text: {input} {choice} Answer:"
        # format prompt without demonstration
        prompt = f"\{INSTRUCTION} {anno_prompt} Text: {input} {choice} Answer:"
    
    return prompt_demo if use_demo else prompt


# def compute_seq_length_dist(split_data, split, answer_type, use_demo):
#     lengths = [] 
#     for sample in tqdm(split_data):
#         full_prompt = generate_prompt(sample, answer_type, use_demo)
#         full_prompt_and_response = full_prompt + " " + sample["output"]
#         length = len(full_prompt_and_response.split(" "))
#         lengths.append(length)

#     # print(f"\n Plot the distribution of sequence lengths in {split} set.")
#     print(f"max sequence length in {split} set is {max(lengths)}.")
#     # plt.hist(lengths)#, bins=50)
#     # plt.show()
#     print(lengths)


def tokenize(tokenizer, sent, add_eos_token = True):
    result = tokenizer(sent) #, return_tensors=None)
    if add_eos_token:
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    return result


def truncate_full_prompt_and_response(
    tokenizer: CustomLlamaTokenizer, 
    full_prompt: str, 
    response: str, 
    truncate_mode: str, # 'first' or 'last'
    max_length: int,
    split: str,
    ):
    """truncate full_prompt_and_response from the beginning of the inputs.
       ensure the length of full_prompt_and_response <= maximum length (including <bos> and <eos>).

       p.s. no truncation for test dataset
    """

    do_truncate = False
    # concat tokenized prompt and response, and remove first <bos> token for both before concatenation
    encoded_full_prompt_and_response_ids = full_prompt["input_ids"][1:] + response["input_ids"][1:]
    encoded_full_prompt_and_response_masks = full_prompt["attention_mask"][1:] + response["attention_mask"][1:]
    
    n_tokens = len(encoded_full_prompt_and_response_ids) + 1  # number of full_prompt_and_response after tokenization/encoding
    response_length = len(response["input_ids"][1:])

    # truncate tokenized full_prompt_and_response from the beginning/end if its length > max_length
    if len(encoded_full_prompt_and_response_ids) > max_length-1: 
        if truncate_mode == "first":
            truncated_full_prompt_and_response_ids = [tokenizer.bos_token_id] + encoded_full_prompt_and_response_ids[-max_length+1:]
            truncated_full_prompt_and_response_masks = [1] + encoded_full_prompt_and_response_masks[-max_length+1:]
        elif truncate_mode == "last":
            truncated_full_prompt_and_response_ids = full_prompt["input_ids"][:max_length-response_length] + response["input_ids"][1:]
            truncated_full_prompt_and_response_masks = full_prompt["attention_mask"][:max_length-response_length] + response["attention_mask"][1:]
        else:
            raise ValueError("invalid truncation mode (first, last are allowed).")

        n_truncated_tokens = n_tokens - len(truncated_full_prompt_and_response_ids)   # number of truncated tokens compared to max_length
        do_truncate = True
    else: 
        truncated_full_prompt_and_response_ids = [tokenizer.bos_token_id] + encoded_full_prompt_and_response_ids
        truncated_full_prompt_and_response_masks = [1] + encoded_full_prompt_and_response_masks
        n_truncated_tokens = 0

    # remove encoded response tokens from the end of full_prompt_and_response if only need tokenized prompt
    truncated_full_prompt =  {"input_ids": truncated_full_prompt_and_response_ids[:-response_length],
                              "attention_mask": truncated_full_prompt_and_response_masks[:-response_length]}

    truncated_full_prompt_and_response = {"input_ids": truncated_full_prompt_and_response_ids, 
                                          "attention_mask": truncated_full_prompt_and_response_masks}

    return truncated_full_prompt, truncated_full_prompt_and_response, n_truncated_tokens, do_truncate


def prepare_sample(
    sample: dict, 
    tokenizer: CustomLlamaTokenizer, 
    max_length: int, 
    alpaca_format: str,
    model_name: str, 
    answer_type: str,
    split: str,
    truncate_mode: str,
    label_name: str,
    use_demo: bool,
    ) -> dict:
    """Processes a single sample.
    
    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - demonstration (optional): examples of input + output
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for supervised training. 
    The input text is formed as a single message including all the instruction, the input and the response.
    The label/target is the same message but can optionally have the instruction + input text masked out.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """

    if label_name == "og":
        response = sample["output"]
    elif label_name == "re1-single":
        if split == "test":
            response = sample["output_re"]
        else:
            response = sample["output_re_1"]
    elif label_name == "re0-single":
        if split == "test":
            response = sample["output_re"]
        else:
            response = sample["output_re_0"]
    elif label_name == "single" or label_name == "single-short" or label_name == "single-x":
        response = sample["output_anno"]
    
    if answer_type=="helm":
        response = response[0]

    full_prompt = generate_prompt(sample, model_name, answer_type, use_demo, label_name)
    full_prompt_and_response = full_prompt + " " + response

    encoded_full_prompt = tokenize(tokenizer, full_prompt, add_eos_token=False)
    encoded_response = tokenize(tokenizer, response, add_eos_token=True)

    truncated_full_prompt, truncated_full_prompt_and_response, n_truncated_tokens, do_truncate = (
        truncate_full_prompt_and_response(tokenizer, 
                                            encoded_full_prompt, 
                                            encoded_response, 
                                            truncate_mode=truncate_mode,
                                            max_length=max_length,
                                            split=split)
        )

    full_prompt_length = len(truncated_full_prompt["input_ids"]) 
    
    # The labels are the full prompt with response, but with the prompt masked out
    labels = truncated_full_prompt_and_response["input_ids"].copy()
    # print(len(labels))
    # print(labels)
    labels[:full_prompt_length] = [IGNORE_INDEX]*full_prompt_length
    # print(len(labels))
    # print(labels)


    # inputs are masked in labels for all splits
    if split == "train":
        return {**sample, 
                "prompt": full_prompt_and_response,
                "input_ids": truncated_full_prompt_and_response["input_ids"], 
                "attention_mask": truncated_full_prompt_and_response["attention_mask"], 
                "labels": labels,
                "do_truncate": do_truncate,
                "truncate_mode": truncate_mode,
                "n_truncated_tokens": n_truncated_tokens,
                } 

    elif split == "val":
        return {**sample, 
                "prompt": full_prompt_and_response,
                "input_ids": truncated_full_prompt_and_response["input_ids"], 
                "attention_mask": truncated_full_prompt_and_response["attention_mask"], 
                "labels": labels,
                "do_truncate": do_truncate,
                "truncate_mode": truncate_mode,
                "n_truncated_tokens": n_truncated_tokens,
                }

    elif split == "test":
        return {**sample, 
                "prompt": full_prompt,
                "input_ids": encoded_full_prompt["input_ids"], #truncated_full_prompt["input_ids"], 
                "attention_mask": encoded_full_prompt["attention_mask"], #truncated_full_prompt["attention_mask"],
                "labels": labels,
                }


def compute_truncation_loss(prepared_data: list, max_length: int):
    # list of truncated numbers from samples that have been truncated 
    # clean means remove non-truncated samples
    n_truncated_tokens_list = [data["n_truncated_tokens"] for data in prepared_data]
    n_truncated_tokens_list_clean = [data["n_truncated_tokens"] for data in prepared_data if data["n_truncated_tokens"] != 0]
    print(n_truncated_tokens_list_clean)
    print(f"Mode value of truncated tokens among truncated samples: {mode(n_truncated_tokens_list_clean)}")
    print(f"Median value of truncated tokens among truncated samples: {median(n_truncated_tokens_list_clean)}")

    # ratio -- truncated samples / total sample
    n_truncated_sample_ratio = len(n_truncated_tokens_list_clean) / len(prepared_data) 
    print(f"Number of truncated samples: {len(n_truncated_tokens_list_clean)}")
    print(f"Truncated sample ratio is : {n_truncated_sample_ratio}")   

    # truncation loss among truncated samples/all samples
    truncated_loss_list = [n/max_length for n in n_truncated_tokens_list]
    truncated_loss_list_clean = [n/max_length for n in n_truncated_tokens_list_clean]
    print(f"Average truncated loss among all samples: {mean(truncated_loss_list)}")
    print(f"Average truncated loss among truncated samples: {mean(truncated_loss_list_clean)}")


def save_alpaca_data(prepared_data: list, save_dir: Path, split: str):
    os.makedirs(save_dir, exist_ok=True)
    # torch.save(prepared_data, Path(save_dir, split+".pt"))
    # print(f"Prepared alpaca {split} data save to {str(save_dir)}/{split}.pt")

    with open(Path(save_dir, split+".json"), "w") as f:
        json.dump(prepared_data, f)
    print(f"Prepared alpaca {split} data save to {str(save_dir)}/{split}.json")


def main(
    datasets: str = "mix-single", #,guest",
    formats: str = "alpaca-option2", #"alpaca-option,alpaca-label",
    model_name: str = "llama2",
    answer_type: str = "harness", #helm", #harness,
    max_length: int = 256,
    truncate_mode: str = "last",
    label_name: str = "og",
    ) -> None:
    """Process original datasets into alpaca-like format and save into json files.

    Args:
        datasets: The datasets to use as a comma separated string
        formats: Data formats to be processed - a comma separated string
        model_name: Pre-trained model to load
    """

    if model_name == "llama2":
        tok = CustomLlamaTokenizer
    elif model_name == "llama3":
        tok = LlamaTokenizer
    elif model_name[:6] == "flant5":
        tok = T5Tokenizer
    else:
        tok = AutoTokenizer

    print(f"*** Loading tokenizer -- {tok}")
    # tokenizer = CustomLlamaTokenizer.from_pretrained(MODEL_NAMES[model_name])
    tokenizer = tok.from_pretrained(MODEL_NAMES[model_name])
    if model_name == "llama3":
        tokenizer.add_special_tokens({"pad_token":"<pad>"}) 
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"

    for ds_name in datasets.split(","):
        for fm_name in formats.split(","):
            for ans in answer_type.split(","):
                data_root_dir = "data/" 
                data_dir = f"{data_root_dir}{ds_name}-{fm_name}"
                feature_dir = f"feature-{label_name}-{truncate_mode}/" 
                save_feature_dir = f"{feature_dir}{ds_name}-{fm_name}-{ans}-{model_name}"
                os.makedirs(save_feature_dir, exist_ok=True)
                
                print(f"\n##### loading {ds_name} dataset in {fm_name} style from {data_dir} -- answer type: {ans}")
                splits = ["train", "val", "test"] 


                for split in splits:
                    with open(Path(data_dir, split+".json"), "r") as file:
                        split_data = json.load(file)
                    print(f"{split} has {len(split_data)} samples.")

                    print(f"Processing {split} split without demostrations...")
                    prepared_data = [prepare_sample(sample, 
                                                    tokenizer, 
                                                    max_length, 
                                                    fm_name, 
                                                    model_name, 
                                                    ans, 
                                                    split, 
                                                    truncate_mode,
                                                    label_name,
                                                    use_demo=False) 
                                        for sample in tqdm(split_data)]

                    print(f"Processing {split} split with demostrations...")
                    prepared_data_and_demo = [prepare_sample(sample, 
                                                            tokenizer, 
                                                            max_length, 
                                                            fm_name, 
                                                            model_name, 
                                                            ans, 
                                                            split, 
                                                            truncate_mode,
                                                            label_name,
                                                            use_demo=True) 
                                                for sample in tqdm(split_data)]
                    

                    save_alpaca_data(prepared_data_and_demo, save_feature_dir+"-demo", split)
                    save_alpaca_data(prepared_data, save_feature_dir, split)

                    print(prepared_data_and_demo[0])
                    print("\n---\n")
                    print(prepared_data[0])
                    print("=============\n")
                    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama2', help="pre-trained model used")
    parser.add_argument("--truncate_mode", type=str, default='last', choices=["first","last"], help="truncation mode to truncate tokenised sentences from beginning or end")
    parser.add_argument("--label_name", type=str, default='og', help="majority/single label type")
    parser.add_argument("--max_length", type=int, default=512, help="maximun sequence length")
    args = parser.parse_args()
    
    main(model_name=args.model_name,
        truncate_mode=args.truncate_mode,
        label_name=args.label_name,
        max_length=args.max_length)