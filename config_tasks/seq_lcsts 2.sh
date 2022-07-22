EXPERIMENT_NAME=${MODEL_TYPE}-lcsts
TASK_NAME=lcsts
DATA_PATH="/mnt/yrfs/litianjian/datasets/LCSTS2.0/DATA"

TRAIN_ARGS="--epochs 30 \
            --batch-size 64 \
            --lr 1e-4 \
            --lr-decay-style linear \
            --warmup 0 \
            --weight-decay 0 \
            --label-smoothing 0.1"

COMMON_ARGS="--save-interval 1000 \
             --log-interval 50 \
             --eval-interval 1000 \
             --eval-iters 100"

TASK_ARGS="--src-seq-length 384 \
           --tgt-seq-length 110 \
           --min-tgt-length 55 \
           --length-penalty 0.7 \
           --no-repeat-ngram-size 3 \
           --num-beams 5 \
           --select-topk \
           --eval-batch-size 4"
