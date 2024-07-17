"""
https://github.com/tloen/alpaca-lora/blob/main/generate.py
https://huggingface.co/docs/transformers/internal/generation_utils#transformers.generation.BeamSearchDecoderOnlyOutput
https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationMixin.compute_transition_scoreshttps://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationMixin.compute_transition_scores
https://stackoverflow.com/questions/76397904/generate-the-probabilities-of-all-the-next-possible-word-for-a-given-text
"""
import os
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import List
import math
from datasets import Dataset, Features, Value, Sequence
import torch
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM, GenerationConfig 
from transformers import AutoModelForCausalLM
from peft import PeftModelForCausalLM, LoraConfig, get_peft_model
# from tokenizer import CustomLlamaTokenizer
from huggingface_hub import login
# hub_token = "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"
hub_token = "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"

login(token=hub_token)


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"  # solve cuda out of memory for model.generate()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAMES = {
                  "llama": "decapoda-research/llama-7b-hf",
                  "llama2-chat": "meta-llama/Llama-2-7b-chat-hf",
                  "llama2": "meta-llama/Llama-2-7b-hf",
                  "llama2-13b": "meta-llama/Llama-2-13b-hf",
                  "llama2-70b": "meta-llama/Llama-2-70b-hf",
                  "llama3": "meta-llama/Meta-Llama-3-8B",
                  "flant5": "google/flan-t5-xl",
                  "mistral": "mistralai/Mistral-7B-v0.3",
              }  

CHOICES = ["A. GBV", "B. Not GBV"]

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
})


def get_dirs(
    alpaca_format: str, 
    dataset: str, 
    feature_name: str,
    model_name: str, 
    answer_type: str, 
    use_demo: bool, 
    ckpt_dir: str, 
    result_dir: str,
    ):

    # feature_root_dir = f"feature-{feature_name[5:]}" 
    model_root_dir = f"model/{model_name}-{feature_name}" 
    result_root_dir = f"{result_dir}/result-{model_name}-{feature_name}/{ckpt_dir}"
    
    name_single = ( 
            f"{dataset}-{alpaca_format}-{answer_type}-{model_name}-demo" if use_demo 
            else f"{dataset}-{alpaca_format}-{answer_type}-{model_name}"
            )
    # test_dataset_dir = Path(feature_root_dir, name_single, "test.json")
    
    full_single_features = ["full-single-first", "full-single-last", "full-single-short-first", "full-single-short-last"]
    # single_features = ["single-first", "single-last", "single-short-first", "single-short-last"]
    single_features = ["single-first", "single-last", "single-short-first", "single-short-last", "single-x-first", "single-x-last"]
    if feature_name in full_single_features: 
        feature_root_dir = f"feature-{feature_name[5:]}" 
        test_dataset_dir = Path(feature_root_dir, name_single, "test.json")

        name = ( 
            f"mix-full-single-{alpaca_format}-{answer_type}-{model_name}-demo" if use_demo 
            else f"mix-full-single-{alpaca_format}-{answer_type}-{model_name}"
            )
        output_dir = Path(result_root_dir, name)
        peft_model_dir = (
            Path(model_root_dir, name) if ckpt_dir=="best"
            else Path(model_root_dir, name, ckpt_dir)
            )
    
    elif feature_name in single_features:
        feature_root_dir = f"feature-{feature_name}" 
        test_dataset_dir = Path(feature_root_dir, name_single, "test.json")

        output_dir = Path(result_root_dir, name_single)
        peft_model_dir = (
            Path(model_root_dir, name_single) if ckpt_dir=="best"
            else Path(model_root_dir, name_single, ckpt_dir)
            )
        
    return test_dataset_dir, peft_model_dir, output_dir


def load_dataset(data_dir):
    # data = Dataset.from_list(torch.load(data_dir))
    with open(data_dir, "r") as file:
        loaded_data = json.load(file)
    for d in loaded_data:
        del d['anno_info']
    data = Dataset.from_list(loaded_data, features=schema)

    return data


def get_prob_list(prob, top_k_prob, tokenizer) -> List[str]:
    # get the top k next token indexes
    topk_ids = torch.topk(prob, top_k_prob).indices.tolist()
    # print(topk_ids)  # [[13, 2, 29896, 29900, 29906, 313, 903, 960], [29909, 4136, 2385, 6466, 3644, 29933, 1576, 29902]]

    # filter the token probabilities for the top k candidates
    topk_probs = [prob[i][topk_ids[i]].tolist() for i in range(prob.shape[0])]

    # decode the top k ids back to words
    topk_tokens = [tokenizer.decode(topk_ids[i]) for i in range(prob.shape[0])]
    # print(topk_tokens)  # [['', '</s>', '1', '0', '2', '(', '_', 'If'], ['A', '####', 'Class', 'Output', 'If', 'B', 'The', 'I']]

    token_prob_list = [list(zip(topk_tokens[i], topk_probs[i])) for i in range(prob.shape[0])]

    return token_prob_list


# def left_padding(input_ids: List[List[int]], max_seq_length: int) -> torch.Tensor():
#     """Padding left for input ids with different lengths
#     """
#     return torch.Tensor([[0]*(max_seq_length-len(x)) + x for x in input_ids]).long() 


def process_inp(input_ids: List[List[int]], max_seq_length: int) -> torch.Tensor():
    """Padding left (len(inp) < max_seq_length) or truncate head (len(inp) >= max_seq_length) for input ids with different lengths
        -> avoid memory error: out of cuda memory
    """
    processed_input_ids = []
    for inp in input_ids:
        processed_input_id = inp[-max_seq_length:] if len(inp) >= max_seq_length else [0]*(max_seq_length-len(inp)) + inp
        processed_input_ids.append(processed_input_id)
    return torch.Tensor(processed_input_ids).long() 
 

def save_gen_results(output_dir, generated_results):
    # save to json file
    save_json_dir = Path(output_dir, "generated_results.json")
    # save to tsv file
    save_tsv_dir = Path(output_dir, "generated_results.tsv")

    if os.path.exists(save_json_dir):
        with open(save_json_dir, "r") as file:
            saved_results = json.load(file)
        concat_results = saved_results + generated_results
    else:
        concat_results = generated_results
    
    with open(save_json_dir, "w") as f:
        json.dump(concat_results, f)
    
    df_generated_results = pd.DataFrame(concat_results)
    df_generated_results.to_csv(save_tsv_dir, sep="\t", header=True, index=False)
    print(f"Generated {len(concat_results)} results are saved to {str(save_json_dir)} & {str(save_tsv_dir)}")


def main(
    dataset: str,
    alpaca_format: str,
    model_name: str,
    use_demo: bool,
    answer_type: str,
    ckpt_dir: str,
    feature_name: str,
    result_dir: str,
    temperature: float = 1.0,
    top_p: float = 0.75,
    top_k: int = 20,
    top_k_prob: int = 10,
    max_new_tokens: int = 2,  #2 for option / 4 for label
    max_seq_length: int = 256,
    n_sample_per_gen: int = 50,
    i_round: int = 0,
    return_dict_in_generate: bool = True,
    output_scores: bool = True,
    output_attentions: bool = True,
    ) -> None:
    """Process original datasets into alpaca-like format and save into json/tsv files.
    """

    test_dataset_dir, peft_model_dir, output_dir = get_dirs(alpaca_format, 
                                                                dataset, 
                                                                feature_name,
                                                                model_name, 
                                                                answer_type, 
                                                                use_demo,  
                                                                ckpt_dir,
                                                                result_dir)

    print(f"\n#### Loading {dataset}'s test dataset in {alpaca_format} format from {str(test_dataset_dir)}")    
    test_data = load_dataset(test_dataset_dir)


    # print(f"\n#### Loading llama model - {MODEL_NAMES[model_name]}")
    # tokenizer = CustomLlamaTokenizer.from_pretrained(MODEL_NAMES[model_name])
    # tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    # tokenizer.bos_token_id = 1
    # tokenizer.eos_token_id = 2
    # tokenizer.padding_side = "left"
    # model = LlamaForCausalLM.from_pretrained(MODEL_NAMES[model_name],
    #                                         torch_dtype=torch.float16,
    #                                         device_map='auto', 
    #                                         load_in_8bit=True, # convert the loaded model into mixed-8bit quantized model (need to install bitsandbytes)
    #                                         llm_int8_enable_fp32_cpu_offload=True, # split model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU
    #                                         )
    
    # print(f"\n#### Loading fine-tuned alpaca-lora model - {peft_model_dir}")
    # model = PeftModelForCausalLM.from_pretrained(model, peft_model_dir)
    # model.print_trainable_parameters()
    # model.to(DEVICE)

    print(f"\n#### Loading model - {MODEL_NAMES[model_name]}")
    # if model_name == "llama2":
    #     tokenizer = CustomLlamaTokenizer.from_pretrained(MODEL_NAMES[model_name])
    #     tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    #     tokenizer.bos_token_id = 1
    #     tokenizer.eos_token_id = 2
    #     tokenizer.padding_side = "left"
    # else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[model_name])
    tokenizer.add_special_tokens({"pad_token":"<pad>"}) 
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"


    model = LlamaForCausalLM.from_pretrained(MODEL_NAMES[model_name],
                                        torch_dtype=torch.float16,
                                        device_map='auto', 
                                        load_in_8bit=True, # convert the loaded model into mixed-8bit quantized model (need to install bitsandbytes)
                                        llm_int8_enable_fp32_cpu_offload=True, # split model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU
                                        )

    print(f"\n#### Loading alpaca-lora model - {peft_model_dir}")
    # model = PeftModelForCausalLM.from_pretrained(model, peft_model_dir)
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj","v_proj"]
    )
    model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()
    model.to(DEVICE)

    # unwind broken decapoda-research config
    model.config.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    print("\n#### Evaluating ...")
    model.eval()
    model = torch.compile(model)
    generation_config = GenerationConfig.from_pretrained(
        MODEL_NAMES[model_name],
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    print("\n#### Generating responses ...")
    max_seq_length = 512 #max([len(x) for x in test_data["input_ids"]]) # for non-truncated test data
    input_ids = process_inp(test_data["input_ids"], max_seq_length).to(DEVICE) #.half() # pad left or truncate for input ids and convert it to half precision
    max_new_tokens = 5 if answer_type=="harness" else 1
    print(input_ids.size())
    import statistics
    print(statistics.mean([input_ids[i].size()[0] for i in range(input_ids.size()[0])]))
    
    # if directly generate all test data -> error: out of cuda memory
    # now generate 50 instances at once is ok
    # allocate all data into different round for generation
    n_round = math.ceil(input_ids.shape[0]/n_sample_per_gen)
    print(f"Number of round is {i_round}")
    if i_round >= n_round-1:
        id_range = [(n_round-1)*n_sample_per_gen, input_ids.shape[0]]
    else:
        id_range = [i_round*n_sample_per_gen, (i_round+1)*n_sample_per_gen]
    sub_input_ids = input_ids[id_range[0]:id_range[1], :]
    print(f"Generating responses [{id_range[0]}, {id_range[1]}] from test dataset.")


    with torch.no_grad():
        # outputs keys = ['sequences', 'sequences_scores', 'scores', 'beam_indices', 'attentions']
        outputs = model.generate(
            input_ids=sub_input_ids,
            generation_config=generation_config,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=output_scores,
            output_attentions=output_attentions,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )


    generated_ids = outputs.sequences[:, max_seq_length:] # only keep generated text after "####Response:"
    # probs = torch.stack(outputs.scores, dim=1).softmax(-1).cpu()  # Get the token probabilities for all candidates 
                                                                  # shape (num_return_sequences, max_new_token, config.vocab_size) = ([10, 2, 32000]) <- outputs.scores shape = (2, 10, 32000)

    # generate response and token/prob list per sequence
    generated_results = []
    # for i in range(outputs.sequences.shape[0]):
    for i, j in zip(range(outputs.sequences.shape[0]), range(id_range[0],id_range[1])):
        generated_label_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True, skip_punctuation_tokens=True)
        # generated_label_probs = get_prob_list(probs[i], top_k_prob, tokenizer)

        pred = {"id": test_data[j]["id"],
                "text": test_data[j]["input"],
                "generated_label": generated_label_text,
                "gold_label": test_data[j]["output"],
                # "generated_probs": generated_label_probs, 
                }

        generated_results.append(pred)
        print(pred["id"])
        print(f'generated content:\n{generated_label_text}')
        # print(generated_label_probs)
        print('#########\n')
    
    # save generated results
    os.makedirs(output_dir, exist_ok=True)
    save_gen_results(output_dir, generated_results)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='mix-single', help="gbv dataset")
    parser.add_argument("--alpaca_format", type=str, default='alpaca-option2', choices=["alpaca-option2"], help="alpaca instruction format")
    parser.add_argument("--model_name", type=str, default='llama3', help="pre-trained model used")
    parser.add_argument("--use_demo", action="store_true", help="whether or not use prompt with demonstrations")
    parser.add_argument("--answer_type", type=str, default='harness', choices=["harness","helm"], help="evaluation format")
    parser.add_argument("--n_sample_per_gen", type=int, default=1, help="number of samples per generation")
    parser.add_argument("--i_round", type=int, default=0, help="number of generation round")
    parser.add_argument("--feature_name", type=str, default='single-first', help="processed feature name")
    parser.add_argument("--result_dir", type=str, default='result-icl', help="directory to save results")
    parser.add_argument("--ckpt_dir", type=str, default='best', help="directory to saved checkpoints")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value for generation")


    args = parser.parse_args()

    main(dataset=args.dataset, 
        alpaca_format=args.alpaca_format, 
        model_name=args.model_name, 
        use_demo=args.use_demo,
        answer_type=args.answer_type,
        n_sample_per_gen=args.n_sample_per_gen, 
        i_round=args.i_round,
        feature_name=args.feature_name,
        result_dir=args.result_dir,
        ckpt_dir=args.ckpt_dir,
        temperature=args.temperature,
        )