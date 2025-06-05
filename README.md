# Setup
```
git clone --recurse-submodules https://github.com/LumiOpen/poro2-scripts-dev.git
cd poro2-scripts-dev
git apply patches/poro2-scripts.patch
```
# Tokenize data with HF tokenizer
Input to the tokenization is jsonl files with field "text" containing the text to be tokenized
```
sbatch preprocess_jsonl.sh <path_to_jsonl_file> <output_folder>
```
# Merge multiple tokenized files in one folder
```
sbatch merge_data.sh <input_folder_with_jsonl_files> <output_folder> <output_file_name>
```

# Convert HF model to Megatron checkpoint
Input the distributed arguments and locations of the original model, tokenizer to script first
```
sbatch scripts/convert_llama3.1-8B_hf_to_meg.sh 
```

# Train model
Change the script to reflect your data and checkpoint locations and training hyperparams
```
sbatch pretrain-poro2-sbatch.sh MODEL_SIZE=8B
```

# Convert the final megatron checkpoint to HF compatible format
```
sbatch scripts/convert_llama3.1-8B_meg_to_hf.sh <input checkpoint> <output_folder>
e.g.
sbatch scripts/convert_llama3.1-8B_meg_to_hf.sh megatron-checkpoints/llama3.1-8B-TP-2-PP-1/iter_0000040/ ./converted_checkpoints
```
