#!/bin/bash
#SBATCH (Add your cluster options here) 

mkdir -p logs

cd ../clinic-bhc-summ-main/src

# The settings here correspond to the logic in run_peft.py. Set according to the situation
MODEL="qwen2.5-14b"
CASE_ID=102
TARGET_EPOCH="19"

echo "Running inference on 200 samples..."
echo "Model: $MODEL, Case: $CASE_ID"

python run_peft_confidence.py \
    --model $MODEL \
    --case_id $CASE_ID

echo "Job completed at: $(date)"