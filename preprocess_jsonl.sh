#!/bin/bash
#SBATCH --job-name=preprocess_jsonl
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=project_462000353
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --mem=256G
#SBATCH --partition=small


module purge
ml use /appl/local/csc/modulefiles/
ml pytorch/2.4

# Input path (update this or pass it as a parameter)
INPUT_FILE=$1
OUTPUT_FOLDER=$2
OUTPUT_FILE=$OUTPUT_FOLDER/$(basename $INPUT_FILE)
mkdir -p $OUTPUT_FOLDER

echo $INPUT_FILE
echo $OUTPUT_FOLDER
echo $OUTPUT_FILE

srun python ./Megatron-LM/tools/preprocess_data.py \
    --input $INPUT_FILE \
    --output $OUTPUT_FILE \
    --json-keys text \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /scratch/project_462000353/models/llama31-8b \
    --append-eod \
    --log-interval 50000 \
    --workers $SLURM_CPUS_PER_TASK;



