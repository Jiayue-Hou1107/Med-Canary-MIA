#!/bin/bash
#SBATCH (Add your cluster options here) 

mkdir -p logs

module purge

export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cusparse/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/torch/lib:$LD_LIBRARY_PATH

echo "Final LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# 4. 准备训练
cd ../clinic-bhc-summ-main/src

# The settings here correspond to the logic in train_peft.py. Set according to the situation
MODEL="qwen2.5-14b"
CASE_ID=102 
NUM_GPUS=2

echo "Configuration:"
echo "Model: $MODEL"
echo "Case ID: $CASE_ID"
echo "Number of GPUs: $NUM_GPUS "
i

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_peft.py \
    --model $MODEL \
    --case_id $CASE_ID \
    --gpu_id 0

echo "Training completed at: $(date)"