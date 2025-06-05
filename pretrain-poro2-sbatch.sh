#!/bin/bash
#SBATCH --job-name=test-rocm-6.2-new-cxi
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=16
#SBATCH --mem=480G
##SBATCH --partition=standard-g
##SBATCH --time=2-00:00:00
#SBATCH --partition=dev-g
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --account=project_462000353
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -eox pipefail
echo "Starting bash script"
module purge
module load LUMI/24.03 partition/G


ln -sf ${SLURM_JOB_NAME}-${SLURM_JOBID}.out logs/latest.out
ln -sf ${SLURM_JOB_NAME}-${SLURM_JOBID}.err logs/latest.err

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CC=gcc-12
export CXX=g++-12

#DISTRIBUTED ARGS
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9998
export WORLD_SIZE=$SLURM_NTASKS #This is valid only if ntasks==ngpus
export CUDA_DEVICE_MAX_CONNECTIONS=1 #This is needed for sequence paralellism


export OMP_NUM_THREADS=1
export HSA_ENABLE_SDMA=0
export PYTHONWARNINGS=ignore
export NVTE_FLASH_ATTN=1

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

DATA_ROOT="data/merged"
CACHE_PATH="${DATA_ROOT}/index-cache"
DATA_PATH="1.0 ${DATA_ROOT}/merged-test-data"


TOKENIZER_MODEL="/scratch/project_462000353/models/llama31-8b"
TENSORBOARD_PATH="tensorboard/$SLURM_JOB_NAME"


if [ "$MODEL_SIZE" = "8B" ]; then
    NHIDDEN=4096
    FFN_HIDDEN_SIZE=14336
    NLAYERS=32
    NHEADS=32
    NUM_KV_HEADS=8
    NUM_QUERY_GROUPS=8
    TIE_WORD_EMBEDDINGS=0
    PP_SIZE=1
    TP_SIZE=2

    SAVE_CKPT_PATH=megatron-checkpoints/llama3.1-8B-TP-2-PP-1
    mkdir -p $SAVE_CKPT_PATH
    LOAD_CKPT_PATH=megatron-checkpoints/llama3.1-8B-TP-2-PP-1

elif [ "$MODEL_SIZE" = "70B" ]; then
    NHIDDEN=8192
    FFN_HIDDEN_SIZE=28672
    NLAYERS=80
    NHEADS=64
    NUM_KV_HEADS=8
    NUM_QUERY_GROUPS=8
    TIE_WORD_EMBEDDINGS=0
    PP_SIZE=8
    TP_SIZE=8

    SAVE_CKPT_PATH=megatron-checkpoints/llama3.1-70B-TP-8-PP-8
    mkdir -p $SAVE_CKPT_PATH
    LOAD_CKPT_PATH=megatron-checkpoints/llama3.1-70B-TP-8-PP-8

else
    echo "Unknown model size"
    exit 1
fi

SEQ_LEN=8192
GLOBAL_BATCH_SIZE=512
GBS_TOKENS=$((GLOBAL_BATCH_SIZE*SEQ_LEN))

#MBS max 1-2 with the default setup                        
MICRO_BATCH_SIZE=1
TOTAL_TOKENS=50_000_000_000
TOTAL_TOKENS=${TOTAL_TOKENS//_}    # drop "_" for bash math

LR_DECAY_ITERS=$((TOTAL_TOKENS/GBS_TOKENS))
TRAIN_ITERS=$LR_DECAY_ITERS
LR_WARMUP_ITERS=596

LOG_INTERVAL=1
EVAL_STEPS=0
INIT_METHOD_STD=0.00747017

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --use-distributed-optimizer \
    --lr 3e-4 \
    --min-lr 1e-8 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --weight-decay 1.0e-1 \
    --ckpt-format torch \
    --lr-decay-iters $LR_DECAY_ITERS \
    --lr-warmup-iters $LR_WARMUP_ITERS \
    "

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --max-position-embeddings $SEQ_LEN \
    --seq-length $SEQ_LEN \
    --train-iters $TRAIN_ITERS \
    --data-path $DATA_PATH \
    --data-cache-path $CACHE_PATH \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --bf16 \
    --disable-bias-linear \
    --init-method-std $INIT_METHOD_STD \
    --normalization RMSNorm \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --attention-softmax-in-fp32 \
    --position-embedding-type rope \
    --use-rope-scaling \
    --group-query-attention \
    --num-query-groups $NUM_QUERY_GROUPS \
    --distributed-timeout-minutes 20 \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --no-bias-dropout-fusion \
    --no-rope-fusion \
    --eval-iters 1 \
    --use-flash-attn \
    --rotary-base 500000 \
    --context-parallel-size 1 \
    --use-rotary-position-embeddings \
    --accumulate-allreduce-grads-in-fp32 \
    --overlap-grad-reduce \
    --overlap-param-gather \
    "

OUTPUT_ARGS=" \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-throughput \
    --log-progress \
    --log-params-norm \
    --log-interval 1 \
    --dataloader-type single \
    --num-workers 5 \
    "
PARALLEL_ARGS="\
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --sequence-parallel \
"

CHECKPOINT_ARGS=""
CPKT_INTERVAL=20

if [ "$LOAD_CKPT_PATH" != "None" ]; then
    CHECKPOINT_ARGS="
    --load $LOAD_CKPT_PATH \
    --no-load-optim \
    --no-load-rng \
    "
fi
if [ "$SAVE_CKPT_PATH" != "None" ]; then
    CHECKPOINT_ARGS="$CHECKPOINT_ARGS \
    --save $SAVE_CKPT_PATH \
    --save-interval $CPKT_INTERVAL \
    "
fi
CMD=" \
    Megatron-LM/pretrain_gpt.py \
    $GPT_ARGS \
    $OPTIMIZER_ARGS \
    $PARALLEL_ARGS \
    $CHECKPOINT_ARGS \
    $OUTPUT_ARGS \
    $DATA_ARGS \
    "

echo '============='
echo $CMD
echo '============='

#     $PROFILE_ARGS \
c="fe"
# Bind mask for one thread per core
BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"


export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd
# /boot,
export CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif
launcher="$PWD/launcher.sh"
export PWD=(`pwd -P`)

# Avoid conflicts with $HOME/.local
export PYTHONUSERBASE=""

echo "Using --cpu-bind=mask_cpu:$BIND_MASK"
srun --label --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec \
    -B $PWD \
    $CONTAINER \
    $launcher \
    $CMD

echo "END $SLURM_JOBID: $(date)"

singularity exec $CONTAINER python3 tools/throughput.py logs/${SLURM_JOB_NAME}-${SLURM_JOBID}.out
