#!/bin/bash

# Usage: bash eval_policy.sh "0,1,2,3"
GPU_LIST=${1:-"0"}
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

N_EXPS=25

echo "Using GPUs: ${GPUS[@]}"

tasks=(
    "door-open-v2-goal-observable"
    "button-press-topdown-v2-goal-observable"
    "faucet-open-v2-goal-observable"
    "handle-press-v2-goal-observable"
    "shelf-place-v2-goal-observable"
    "hammer-v2-goal-observable"
    "door-close-v2-goal-observable"
    "faucet-close-v2-goal-observable"
    "button-press-v2-goal-observable"
)

RESULT_ROOT="results/"
LOCK_DIR="/tmp/gpu_locks"

mkdir -p "$LOCK_DIR"

# Function to run one task safely on one GPU
run_task() {
    local task=$1
    local gpu_id=$2
    local lock_file="$LOCK_DIR/gpu_${gpu_id}.lock"

    echo "Waiting for GPU $gpu_id to be free for $task"
    flock -w 3600 "$lock_file" bash -c "
        echo \"Starting $task on GPU $gpu_id\"
        CUDA_VISIBLE_DEVICES=$gpu_id xvfb-run -a python eval_my_flow_policy.py \
            --env_name \"$task\" --n_exps $N_EXPS --result_root \"$RESULT_ROOT\"
        echo \"Finished $task on GPU $gpu_id\"
    "
}

# Launch tasks in background, assign GPUs round-robin
i=0
for task in "${tasks[@]}"; do
    gpu_index=$((i % NUM_GPUS))
    gpu_id=${GPUS[$gpu_index]}
    run_task "$task" "$gpu_id" &
    ((i++))
done

wait
echo "All tasks completed."

python <<EOF
import os, json

result_root = "${RESULT_ROOT}"
success_rates_path = os.path.join(result_root, "success_rates")
combined_result = {}

for filename in os.listdir(success_rates_path):
    if filename.endswith("_result.json"):
        env_name = filename.replace("_result.json", "")
        with open(os.path.join(success_rates_path, filename), "r") as f:
            combined_result[env_name] = json.load(f)

with open(os.path.join(result_root, "result_dict.json"), "w") as f:
    json.dump(combined_result, f, indent=4)
EOF