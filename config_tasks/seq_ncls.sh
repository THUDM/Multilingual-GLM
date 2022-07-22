EXPERIMENT_NAME=${MODEL_TYPE}-ncls
TASK_NAME=ncls
DATA_PATH="/mnt/yrfs/litianjian/mGLM/multi-finetune/NCLS-Data/EN2ZHSUM/data"

TRAIN_ARGS="--epochs 10 \
            --batch-size 64 \
            --lr 5e-5 \
            --lr-decay-style linear \
            --warmup 0 \
            --weight-decay 0 \
            --label-smoothing 0.1"

COMMON_ARGS="--save-interval 1000 \
             --log-interval 50 \
             --eval-interval 1000 \
             --eval-iters 100"

TASK_ARGS="--src-seq-length 768 \
           --tgt-seq-length 160 \
           --min-tgt-length 100 \
           --length-penalty 0.7 \
           --no-repeat-ngram-size 3 \
           --num-beams 5 \
           --select-topk \
           --eval-batch-size 4"
