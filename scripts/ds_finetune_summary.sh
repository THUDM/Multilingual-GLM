DATA_ROOT="/mnt/yrfs/litianjian/mGLM/multi-finetune/" #for xtreme datasets

CHECKPOINT_PATH="new_checkpoints"
SAVE_PATH="finetune_checkpoints"
DATESTR=$(date +"%m-%d-%H-%M")

source $1    # Model
source $2    # Task

MP_SIZE=1
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
if [ -z $AVAILABLE_DEVICES ];then
  AVAILABLE_DEVICES=0,1,2,3
fi
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
DISTRIBUTED_ARGS="${OPTIONS_NCCL} deepspeed --master_port ${MASTER_PORT} --include localhost:${AVAILABLE_DEVICES}"

EXPERIMENT_NAME=${EXPERIMENT_NAME}_${DATESTR}
mkdir logs/${TASK_NAME}
run_cmd="${DISTRIBUTED_ARGS} finetune_glm.py \
       --deepspeed \
       --deepspeed_config config_tasks/config_blocklm_large_summary.json \
       --finetune \
       --experiment-name ${EXPERIMENT_NAME} \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${SAVE_PATH} \
       --checkpoint-activations \
       --num-workers 1 \
       --no-load-lr-scheduler \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
       $TASK_ARGS \
       --fp16 \
       --model-parallel-size ${MP_SIZE} \
       --overwrite \
       2>&1 | tee logs/${TASK_NAME}/log-${EXPERIMENT_NAME}.txt"

echo ${run_cmd}
eval ${run_cmd}
