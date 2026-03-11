#!/bin/bash
#SBATCH (Add your cluster options here) 

mkdir -p logs

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/torch/lib:$LD_LIBRARY_PATH

cd ../clinic-bhc-summ-main/src

# Select Model Name
MODELS_TO_TEST=("qwen2.5-14b" "qwen3-8b")

for MODEL_NAME in "${MODELS_TO_TEST[@]}"
do  
    python run_base_confidence.py \
        --model "$MODEL_NAME" \
        --case_id 101 \
        --batch_size 1
done

echo "All baseline evaluations completed."