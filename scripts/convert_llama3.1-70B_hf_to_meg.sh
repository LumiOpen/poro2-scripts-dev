#!/bin/bash

#SBATCH --job-name=convert_llama3
#SBATCH --time=01:00:00
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH --gpus-per-node=mi250:8
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account=project_462000353
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

export NVTE_FLASH_ATTN=1
CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif
HF_FORMAT_DIR=/scratch/project_462000353/models/Llama31_70B
TOKENIZER_MODEL=$HF_FORMAT_DIR
TARGET_PP=8
TARGET_TP=8
MEGATRON_FORMAT_DIR=megatron-checkpoints/llama3.1-70B-TP-$TARGET_TP-PP-$TARGET_PP
export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash,/appl,/usr/lib64/libjansson.so.4,/usr/lib64/libcxi.so.1,/opt/cray,/var/spool/slurmd
export CUDA_DEVICE_MAX_CONNECTIONS=1


# This is a workaround for the issue with the container not having the correct version of transformers
# so we need to install it to userspace
#/bin/bash -c "pip install transformers==4.48.2;
singularity exec $CONTAINER python3 Megatron-LM/tools/checkpoint/convert.py \
    --model-type GPT \
    --loader llama_mistral \
    --model-size llama3-70B \
    --checkpoint-type 'hf' \
    --saver mcore \
    --target-tensor-parallel-size ${TARGET_TP} \
    --target-pipeline-parallel-size ${TARGET_PP} \
    --load-dir ${HF_FORMAT_DIR} \
    --save-dir ${MEGATRON_FORMAT_DIR} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --bf16 \
   

