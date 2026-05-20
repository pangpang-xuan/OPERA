set -x

# nnodes=$1
# node_rank=$2
# master_addr=$3
# master_port=$4
# save_path="checkpoint"

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2
HOME="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/EVA/zhangzijian14/project/verl"

# CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=auto --nnodes=$nnodes --node_rank=$node_rank --master_addr=$master_addr --master_port=$master_port \
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/../data/gsm8k/train.parquet \
    data.val_files=$HOME/../data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-4 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/EVA/zhangzijian14/models/huggingface.co/Qwen/Qwen2.5-0.5B-Instruct \
    model.fsdp_config.model_dtype=bfloat16 \
    trainer.default_local_dir=$HOME/../$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen-2.5-32b-instruct-sp2 \
    trainer.logger=['console'] \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true \
    trainer.total_epochs=5

    # trainer.total_training_steps=1 \
    # trainer.default_hdfs_dir=null $@ \