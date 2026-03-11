import os
import sys
import json
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from scipy import stats

import constants
import parser
import process


def load_all_engines(args):
    print(f"System: Initializing analysis engine for {args.model}...")
    dir_model_peft = os.path.join(args.dir_models_tuned, str(args.target_epoch))
    
    config = PeftConfig.from_pretrained(dir_model_peft)
    base_model_path = constants.MODELS.get(args.model, config.base_model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 

    # Optimize for Blackwell GPU with SDPA
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa" 
    )
    model = PeftModel.from_pretrained(base_model, dir_model_peft)
    return model, tokenizer

def calculate_token_exposure(model, tokenizer, impression_text, target_word, finding_text, args):
    """
    Calculates Probability, Rank, and Exposure for a target word.
    Formula: Exposure = log2(VocabSize) - log2(Rank)
    """
    if not target_word or target_word.lower() not in impression_text.lower():
        return None
    
    prefix_impression = impression_text.lower().split(target_word.lower())[0]

    if args.context_len > 0:
        words = prefix_impression.split()

        prefix_impression = " ".join(words[-args.context_len:])
    
    instruction = constants.START_PREFIX

    # Construct Prompt
    model_key = args.model.lower()
    if "qwen" in model_key:
        prompt = (f"<|im_start|>system\n{instruction}<|im_end|>\n"
                  f"<|im_start|>user\n### Clinical Findings:\n{finding_text}\n\nProvide the Clinical Impression.<|im_end|>\n"
                  f"<|im_start|>assistant\n{prefix_impression}")
    elif "llama3" in model_key:
        prompt = (f"<|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
                  f"<|start_header_id|>user<|end_header_id|>\n\n### Clinical Findings:\n{finding_text}<|eot_id|>"
                  f"<|start_header_id|>assistant<|end_header_id|>\n\n{prefix_impression}")
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Clinical Findings:\n{finding_text}\n\n### Clinical Impression:\n{prefix_impression}"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    target_ids = tokenizer.encode(target_word, add_special_tokens=False)
    if not target_ids: return None
    
    # We probe the first token of the target word for Rank
    gt_token_id = target_ids[0]

    with torch.no_grad():
        outputs = model(inputs.input_ids)
        logits = outputs.logits[:, -1, :] # Last position
        probs = torch.softmax(logits, dim=-1)[0]
        
        # 1. Probability
        prob = probs[gt_token_id].item()
        
        # 2. Rank (1-based index in the sorted list)
        # Count how many tokens have a higher probability than the ground truth
        rank = (probs > probs[gt_token_id]).sum().item() + 1
        
        # 3. Exposure
        vocab_size = tokenizer.vocab_size
        exposure = np.log2(vocab_size) - np.log2(rank)
        
        return {"prob": prob, "rank": rank, "exposure": exposure}

def main():
    parser_temp = argparse.ArgumentParser(add_help=False)
    parser_temp.add_argument('--target_epoch', type=int, default=19)
    parser_temp.add_argument('--num_samples', type=int, default=791)
    parser_temp.add_argument('--data_dir', type=str, default="../clin-bhc-summ/data")
    parser_temp.add_argument('--context_len', type=int, default=-1, help="只保留目标词前的N个词，-1表示全上下文")
    args_temp, remaining = parser_temp.parse_known_args()
    
    sys.argv = [sys.argv[0]] + remaining
    args = parser.get_parser()
    args.target_epoch = args_temp.target_epoch
    args.num_samples = args_temp.num_samples
    args.data_dir = args_temp.data_dir
    args.context_len = args_temp.context_len

    model, tokenizer = load_all_engines(args)
    model.eval()

    results_data = []

    for split in ['train', 'test']:
        json_path = os.path.join(args.data_dir, f"{split}_rare_canaries_index.json")
        if not os.path.exists(json_path): 
            print(f"Warning: Missing {json_path}")
            continue
            
        with open(json_path, 'r', encoding='utf-8') as f:
            canary_data = json.load(f)[:args.num_samples]

        print(f"Processing {split.upper()} set: {len(canary_data)} samples...")
        for item in tqdm(canary_data):
            sample_metrics = {"probs": [], "ranks": [], "exposures": []}
            
            for target in item['rare_targets']:
                res = calculate_token_exposure(model, tokenizer, item['impression'], target, item['finding'], args)
                if res:
                    sample_metrics["probs"].append(res["prob"])
                    sample_metrics["ranks"].append(res["rank"])
                    sample_metrics["exposures"].append(res["exposure"])

            if sample_metrics["exposures"]:
                results_data.append({
                    "split": split,
                    "idx": item['sample_idx'],
                    "findings": item['finding'],
                    "impression": item['impression'],
                    "canaries": "; ".join(item['rare_targets']),
                    "mean_cloze_prob": np.mean(sample_metrics["probs"]),
                    "mean_rank": np.mean(sample_metrics["ranks"]),
                    "mean_exposure": np.mean(sample_metrics["exposures"]),
                    "max_exposure": np.max(sample_metrics["exposures"])
                })

    if not results_data:
        print("Fatal: Execution yielded no results.")
        sys.exit(1)

    df = pd.DataFrame(results_data)
    
    # Statistical Significance Analysis (Exposure Based)
    trn_exp = df[df['split'] == 'train']['mean_exposure'].dropna()
    tst_exp = df[df['split'] == 'test']['mean_exposure'].dropna()
    
    if len(trn_exp) > 1 and len(tst_exp) > 1:
        t_stat, p_val = stats.ttest_ind(trn_exp, tst_exp, equal_var=False)
        print("\n" + "="*50)
        print(f"MIA EXPOSURE REPORT - MODEL: {args.model}")
        print(f"Train Mean Exposure: {trn_exp.mean():.4f}")
        print(f"Test Mean Exposure: {tst_exp.mean():.4f}")
        print(f"Exposure Gap (ΔExp): {trn_exp.mean() - tst_exp.mean():.4f}")
        print(f"P-Value: {p_val:.2e}")
        print("="*50)

    save_path = os.path.join(args.dir_out, f"exposure_mia_results_ep{args.target_epoch}.csv")
    df.to_csv(save_path, index=False)
    print(f"Results with Exposure metrics saved to: {save_path}")

if __name__ == "__main__":
    main()