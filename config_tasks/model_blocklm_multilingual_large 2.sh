MODEL_TYPE="blocklm-large-multilingual"
MODEL_ARGS="--block-lm \
            --task-mask \
            --cloze-eval \
            --num-layers 24 \
            --hidden-size 1536 \
            --num-attention-heads 16 \
            --max-sequence-length 1025 \
            --tokenizer-type ChineseSPTokenizer \
            --tokenizer-model-type /mnt/yrfs/litianjian/mGLM/mglm250k/mglm250k-uni.model \
            --load-pretrained ${CHECKPOINT_PATH}/blocklm-roberta-large-multi04-15-10-06"
