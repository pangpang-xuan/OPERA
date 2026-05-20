
set -x

nvidia-smi

# TORCH DEBUG
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export VLLM_ATTENTION_BACKEND=XFORMERS
# NCCL DEBUG
# export NCCL_DEBUG=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,P2P,NET,GRAPH,ENV,DYNDBG
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=20
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_RETRY_CNT=15
export RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S=3600
export NCCL_PXN_DISABLE=${NCCL_PXN_DISABLE:-1}
echo "====[DEBUG] NCCL_IB_DISABLE=1, NCCL_SOCKET_IFNAME set to: $NCCL_SOCKET_IFNAME ===="


WORKING_DIR=${WORKING_DIR:-"verl"}
RUNTIME_ENV=${RUNTIME_ENV:-"verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-4}

project_name='rethinkingmind_rl_grpo'
exp_name="qwen3_8b_$(date +"%Y-%m-%d_%H-%M")"

# MODEL_PATH=outputs/sft/v18-20260205-235655/checkpoint-6300 # qwen3
# MODEL_PATH=outputs/sft/v41-20260408-114344/checkpoint-6300   # llama3.1-8b
# MODEL_PATH=outputs/sft/v42-20260413-130809/checkpoint-1575 # qwen3-32b
MODEL_PATH=outputs/sft/v18-20260205-235655/checkpoint-6300

TRAIN_FILE=thinkingwriter_training/verl/rl.parquet

TEST_FILE=thinkingwriter_training/verl/rethinkingwriter_test.parquet
train_files="['$TRAIN_FILE']"
test_files="['$TEST_FILE']"

mkdir -p "logs/verl_wandb/${project_name}/${exp_name}"
mkdir -p "logs/tensorboard/${project_name}/${exp_name}"

chmod -R 777 "logs/verl_wandb/${project_name}/${exp_name}"
chmod -R 777 "logs/tensorboard/${project_name}/${exp_name}"

export WANDB_MODE="offline"
export WANDB_API_KEY=""
export WANDB_DIR="logs/verl_wandb/${project_name}/${exp_name}"
export TENSORBOARD_DIR="logs/tensorboard/${project_name}/${exp_name}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"

CKPTS_DIR=checkpoint_verl/checkpoints/${project_name}/${exp_name}
mkdir -p ${CKPTS_DIR}
chmod -R 777 ${CKPTS_DIR}

# connection to all nodes and master
main_output=$(python verl/hope_info.py)
echo $main_output

IFS=' ' read -r main rank <<< "$main_output"
echo $main
echo $rank
timestamp=$(date +%s)

PET_MASTER_PORT=8279

if [ "$rank" -eq 0 ]; then
  ray start --head --port=$PET_MASTER_PORT --min-worker-port=10002 --max-worker-port=10101
  sleep 120

  ray status
  ray job submit \
     --runtime-env="${RUNTIME_ENV}" \
     --working-dir "${WORKING_DIR}" \
     -- python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=64 \
    data.max_prompt_length=6144 \
    data.max_response_length=10240 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.4 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    custom_reward_function.path="verl/verl/utils/reward_score/rethinkingmind.py" \
    custom_reward_function.name="compute_score" \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$exp_name \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=20000 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 $@ 2>&1 | tee "${TENSORBOARD_DIR}/train.log"
  touch ${CKPTS_DIR}/main_done_${main}.txt
  sleep 15 
else
  sleep 10
  ray start --address="$main:$PET_MASTER_PORT"

  while [ ! -f ${CKPTS_DIR}/main_done_${main}.txt ]; do
    echo "Waiting for main node to finish..., no fold: ${CKPTS_DIR}/main_done_${main}.txt"
    sleep 60
  done
  echo "Training finished, worker exiting."
fi
