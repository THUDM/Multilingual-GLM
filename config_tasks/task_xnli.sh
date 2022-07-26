EXPERIMENT_NAME=${MODEL_TYPE}-xnli
TASK_NAME=XNLI
DATA_PATH="${DATA_ROOT}/xnli"
MAX_SEQ_LEN=256

LR_SINGLE=3e-5
EPOCH_SINGLE=10
XXLARGE_EPOCH=10
PROMPT_EPOCH=200

TRAIN_ARGS="--lr-decay-style linear \
            --warmup 0 \
            --weight-decay 1.0e-1 \
            --pattern-id 4"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 100 \
             --eval-iters 100"

#PATTERN_IDS=4 #(0 1 2 3)
PROMPT_IDS=(1 2 3)

BATCH_SIZE=64
