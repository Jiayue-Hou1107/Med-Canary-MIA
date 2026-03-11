import os
import sys
import time
import json
import torch
import argparse
import numpy as np
import string
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoModel, BertTokenizer, BertModel, BertConfig
from peft import PeftModel, PeftConfig
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import constants
import parser
import process

# This section cannot compute BERTScore due to the offline cluster.
LOCAL_BERT_PATH = "../clinic-bhc/clin-bhc-summ/models/ClinicalBERT/"
PROMPT_PREFIX = constants.START_PREFIX 


def calculate_bert_score_manual_offline(preds, refs, model_path, device, batch_size=32):
    print(f"Loading BERT for manual scoring from: {model_path}")
    vocab_path = os.path.join(model_path, "vocab.txt")
    config_path = os.path.join(model_path, "config.json")
    bin_path = os.path.join(model_path, "pytorch_model.bin")
    
    if not os.path.exists(vocab_path) or not os.path.exists(config_path) or not os.path.exists(bin_path):
        print("Warning: BERT files missing. Skipping BERTScore.")
        return 0.0
        
    try:
        tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=True)
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config = BertConfig(**config_dict)
        model = BertModel(config)
        state_dict = torch.load(bin_path, map_location='cpu')
        new_state_dict = {k[5:] if k.startswith('bert.') else k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device).eval()
    except Exception as e:
        print(f"Failed to load BERT model manual: {e}")
        return 0.0

    all_f1_scores = []
    preds_trunc = [" ".join(p.split()[:510]) for p in preds]
    refs_trunc = [" ".join(r.split()[:510]) for r in refs]

    for i in tqdm(range(0, len(preds_trunc), batch_size), desc="Calculating BERTScore"):
        batch_preds = preds_trunc[i:i+batch_size]
        batch_refs = refs_trunc[i:i+batch_size]
        with torch.no_grad():
            enc_preds = tokenizer(batch_preds, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            out_preds = model(**enc_preds)
            emb_preds = out_preds.last_hidden_state / (out_preds.last_hidden_state.norm(dim=-1, keepdim=True) + 1e-9)
            
            enc_refs = tokenizer(batch_refs, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            out_refs = model(**enc_refs)
            emb_refs = out_refs.last_hidden_state / (out_refs.last_hidden_state.norm(dim=-1, keepdim=True) + 1e-9)
            
            for j in range(len(batch_preds)):
                p_mask = enc_preds['attention_mask'][j].bool()
                r_mask = enc_refs['attention_mask'][j].bool()
                p_vecs = emb_preds[j][p_mask] 
                r_vecs = emb_refs[j][r_mask]
                
                if p_vecs.size(0) == 0 or r_vecs.size(0) == 0:
                    all_f1_scores.append(0.0)
                    continue
                
                sim_matrix = torch.matmul(p_vecs, r_vecs.t())
                recall_val = sim_matrix.max(dim=0)[0].mean().item()
                precision_val = sim_matrix.max(dim=1)[0].mean().item()
                f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0.0
                all_f1_scores.append(f1)
                
    return np.mean(all_f1_scores) * 100

def compute_distinct_n(predictions, n):
    if len(predictions) == 0: return 0.0
    distinct_ngrams = set()
    total_ngrams = 0
    for text in predictions:
        tokens = text.strip().split()
        if len(tokens) < n: continue
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        distinct_ngrams.update(ngrams)
        total_ngrams += len(ngrams)
    if total_ngrams == 0: return 0.0
    return len(distinct_ngrams) / total_ngrams

def compute_metrics(refs, preds):
    print("Computing metrics (ROUGE, BLEU-1, Distinct, BERTScore)...")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    bleu_refs, bleu_preds = [], []

    for ref, pred in zip(refs, preds):
        if not ref.strip(): ref = "empty"
        if not pred.strip(): pred = "empty"
        s = scorer.score(ref, pred)
        rouge_scores['rouge1'].append(s['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(s['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(s['rougeL'].fmeasure)
        bleu_refs.append([ref.split()])
        bleu_preds.append(pred.split())

    results = {k: np.mean(v) * 100 for k, v in rouge_scores.items()}
    smoothie = SmoothingFunction().method1
    results['bleu_1'] = corpus_bleu(bleu_refs, bleu_preds, weights=(1.0, 0, 0, 0), smoothing_function=smoothie) * 100
    results['distinct_1'] = compute_distinct_n(preds, 1) * 100
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results['bert_score'] = calculate_bert_score_manual_offline(preds, refs, LOCAL_BERT_PATH, device)
    return results

#  Load Model
def load_model(args):
    model_epoch = args.target_epoch
    dir_model_peft = os.path.join(args.dir_models_tuned, str(model_epoch))
    
    if not os.path.exists(dir_model_peft):
        raise FileNotFoundError(f"Cannot find the weight directory: {dir_model_peft}")
    
    print(f'Loading PEFT Adapter from: {dir_model_peft}')
    config = PeftConfig.from_pretrained(dir_model_peft)
    
    base_model_path = constants.MODELS.get(args.model, config.base_model_name_or_path)

    print(f"Loading Base Model: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 
    
    print("Loading Base Model in float16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        return_dict=True, 
        torch_dtype=torch.float16,
        device_map="auto",       
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, dir_model_peft)
    return model, tokenizer

def run_subset_with_probs(args, model, tokenizer, task_name, num_samples):
    print(f"\n>>> Running Inference on {task_name.upper()} (Sampling {num_samples})")
    raw_dataset = process.load_data(args, task=task_name)
    
    if len(raw_dataset) > num_samples:
        raw_dataset = raw_dataset.shuffle(seed=42).select(range(num_samples))

    is_llama3 = "llama3" in args.model.lower()
    is_qwen = "qwen" in args.model.lower()

    def template_dataset(sample):
        instruction = PROMPT_PREFIX
        finding = sample['sentence']
        # Apply specific inference templates based on model type to ensure consistency with training.
        if is_qwen:
            prompt = f"<|im_start|>system\n{instruction}<|im_end|>\n"
            prompt += f"<|im_start|>user\n### Clinical Findings:\n{finding}\n\nProvide the Clinical Impression.<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n"
        elif is_llama3:
            sys_part = f"<|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
            user_part = f"<|start_header_id|>user<|end_header_id|>\n\n### Clinical Findings:\n{finding}\n\nBased on the findings above, provide the Clinical Impression.<|eot_id|>"
            prompt = sys_part + user_part + f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            prompt = process.causal_formatting_test(sample, instruction)
            
        sample["prompt_text"] = prompt
        return sample

    dataset_for_model = raw_dataset.map(template_dataset)
    data_records = [] 
    data_list = list(dataset_for_model)
    
    for i in tqdm(range(0, len(data_list), args.batch_size), desc=f"{task_name.upper()}"):
        batch_samples = data_list[i : i + args.batch_size]
        batch_prompts = [s['prompt_text'] for s in batch_samples]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=constants.MAX_LEN).to(args.device)
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                do_sample=False, num_beams=1, repetition_penalty=1.1, max_new_tokens=250,      
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                output_scores=True,           
                return_dict_in_generate=True  
            )
            
        generated_sequences = outputs.sequences[:, input_len:]
        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        decoded_outputs = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
        
        for j in range(len(batch_samples)):
            sample_token_ids = generated_sequences[j]
            sample_scores = transition_scores[j]
            
            token_prob_list = []
            confidences = []
            
            for token_id, score in zip(sample_token_ids, sample_scores):
                if token_id == tokenizer.eos_token_id: break
                token_str = tokenizer.decode(token_id)
                prob = np.exp(score.cpu().item())
                token_prob_list.append((token_str, prob))
                confidences.append(prob)
            
            data_records.append({
                "split": task_name,
                "case_id": args.case_id,
                "sample_idx": batch_samples[j]['idx'],
                "prompt": batch_samples[j]['prompt_text'], 
                "reference": batch_samples[j]['text_label'],
                "prediction": decoded_outputs[j],
                "avg_confidence": np.mean(confidences) if confidences else 0.0,
                "token_probabilities": json.dumps(token_prob_list, ensure_ascii=False)
            })

    return data_records

def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    parser_temp = argparse.ArgumentParser(add_help=False)
    parser_temp.add_argument('--target_epoch', type=str, default='19')
    parser_temp.add_argument('--batch_size', type=int, default=1)
    args_temp, remaining_argv = parser_temp.parse_known_args()
    
    sys.argv = [sys.argv[0]] + remaining_argv
    args = parser.get_parser()
    args.target_epoch = args_temp.target_epoch
    args.batch_size = args_temp.batch_size
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.case_id is None: args.case_id = 100 
        
    try:
        model, tokenizer = load_model(args) 
        model.eval()
        
        print(">>> Sampling 100 from TRAIN set...")
        train_records = run_subset_with_probs(args, model, tokenizer, task_name='trn', num_samples=100)
        
        print(">>> Sampling 100 from TEST set...")
        test_records = run_subset_with_probs(args, model, tokenizer, task_name='test', num_samples=100)

        all_records = train_records + test_records
        df = pd.DataFrame(all_records)
        
        # Save Path
        save_path = os.path.join(args.dir_out, f"analysis_case{args.case_id}_ep{args.target_epoch}_probs")
        os.makedirs(save_path, exist_ok=True)
        csv_file = os.path.join(save_path, "inference_with_confidence.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"\nSaved results to: {csv_file}")

        # Calculation Indicators
        def report_metrics(records, split_name):
            if not records: return
            print(f"\n" + "="*20 + f" {split_name} " + "="*20)
            refs = [r['reference'] for r in records]
            preds = [r['prediction'] for r in records]
            m = compute_metrics(refs, preds)
            for k, v in m.items(): print(f"{k:12}: {v:.2f}")

        report_metrics(train_records, "TRAIN (100 samples)")
        report_metrics(test_records, "TEST (100 samples)")
        
    except Exception as e:
        print(f"\n!!! Error !!!\n{e}")
        import traceback; traceback.print_exc()

if __name__ == '__main__':
    main()