set -x
export USE_OPTIMIZED_MODEL=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export USE_FLASH_ATTENTION=0
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
USE_FLASH_ATTENTION=0 torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
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
    trainer.default_local_dir=$HOME/../$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen-2.5-32b-instruct-sp2 \
    trainer.logger=['console'] \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true \
    trainer.total_epochs=5 \
    +trainer.device=npu

#     # trainer.total_training_steps=1 \
#     # trainer.default_hdfs_dir=null $@ \
# model.fsdp_config.model_dtype=bfloat16 \

# set -x

# export VLLM_ATTENTION_BACKEND=XFORMERS
# HOME="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/EVA/zhangzijian14/project/verl"
# python3 -m verl.trainer.main_ppo \
#     algorithm.adv_estimator=grpo \
#     data.train_files=$HOME/../data/gsm8k/train.parquet \
#     data.val_files=$HOME/../data/gsm8k/test.parquet \
#     data.train_batch_size=128 \
#     data.max_prompt_length=512 \
#     data.max_response_length=128 \
#     data.filter_overlong_prompts=True \
#     data.truncation='error' \
#     actor_rollout_ref.model.path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/EVA/zhangzijian14/models/huggingface.co/Qwen/Qwen2.5-0.5B-Instruct \
#     actor_rollout_ref.actor.optim.lr=5e-7 \
#     actor_rollout_ref.model.use_remove_padding=False \
#     actor_rollout_ref.actor.entropy_coeff=0.001 \
#     actor_rollout_ref.actor.ppo_mini_batch_size=64 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
#     actor_rollout_ref.actor.use_kl_loss=True \
#     actor_rollout_ref.actor.kl_loss_coef=0.001 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
#     actor_rollout_ref.rollout.enable_chunked_prefill=False \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
#     actor_rollout_ref.rollout.n=5 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     algorithm.kl_ctrl.kl_coef=0.001 \
#     trainer.critic_warmup=0 \
#     trainer.logger=['console'] \
#     trainer.project_name='verl_grpo_example_gsm8k' \
#     trainer.experiment_name='qwen2_7b_function_rm' \
#     trainer.n_gpus_per_node=4 \
#     trainer.nnodes=1 \
#     trainer.save_freq=-1 \
#     trainer.test_freq=5 \
#     trainer.total_epochs=1 \
#     trainer.device=npu $@