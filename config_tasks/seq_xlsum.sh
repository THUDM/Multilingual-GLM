EXPERIMENT_NAME=${MODEL_TYPE}-xlsum
TASK_NAME=xlsum
DATA_PATH="${DATA_ROOT}/XLSum"

TRAIN_ARGS="--epochs 18 \
            --batch-size 32 \
            --lr 4e-5 \
            --lr-decay-style linear \
            --warmup 0.18 \
            --weight-decay 0 \
            --label-smoothing 0.1"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 1000 \
             --eval-iters 100"

TASK_ARGS="--src-seq-length 608 \
           --tgt-seq-length 160 \
           --min-tgt-length 55 \
           --length-penalty 0.7 \
           --no-repeat-ngram-size 3 \
           --num-beams 5 \
           --select-topk \
           --eval-batch-size 4"
