
MP_SIZE=1
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
if [ -z $AVAILABLE_DEVICES ];then
  AVAILABLE_DEVICES=1
fi
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
DISTRIBUTED_ARGS="${OPTIONS_NCCL} deepspeed --master_port ${MASTER_PORT} --include localhost:${AVAILABLE_DEVICES}"


source $1

MPSIZE=1
MAXSEQLEN=200
MASTER_PORT=$(10086)

#SAMPLING ARGS
TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=40
TOPP=0

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

config_json="$script_dir/ds_config.json"

run_cmd="${DISTRIBUTED_ARGS} --master_port $MASTER_PORT  generate_samples.py \
       --mode inference \
       --model-parallel-size $MPSIZE \
       $MODEL_ARGS \
       --num-beams 35 \
       --no-repeat-ngram-size 3 \
       --length-penalty 0.7 \
       --temperature $TEMP \
       --top_k $TOPK \
       --top_p $TOPP \
       --fp16 \
       --out-seq-length $MAXSEQLEN \
       --batch-size 2 \
       --out-seq-length 200"

echo ${run_cmd}
eval ${run_cmd}


