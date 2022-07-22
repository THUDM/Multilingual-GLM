TASK_NAME="tydiqa"
EXPERIMENT_NAME=${MODEL_TYPE}-${TASK_NAME}
DATA_PATH="/mnt/yrfs/litianjian/mGLM/multi-finetune/xtreme-master/download/tydiqa"

LR_SINGLE=1e-5
EPOCH_SINGLE=20
BATCH_SINGLE=12

TRAIN_ARGS="--lr-decay-style linear \
            --warmup 0.06 \
            --weight-decay 1.0e-1 \
            --label-smoothing 0.1 \
            --epochs 40"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 200 \
             --eval-interval 1000 \
             --eval-iters 100 \
             --eval-epoch 1 \
             --overwrite"

TASK_ARGS="--src-seq-length 512 \
           --tgt-seq-length 64 \
           --min-tgt-length 0 \
           --length-penalty 0 \
           --num-beams 5 \
           --select-topk \
           --eval-batch-size 8 \
           --validation-metric F1"

#           --load /dataset/fd5061f6/finetune_checkpoints/blank-base-squad_v1
#           --load /dataset/fd5061f6/finetune_checkpoints/blocklm-roberta-large-squad_v1
