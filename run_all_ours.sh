#!/bin/bash

# ================= Configuration Area =================
export CUDA_VISIBLE_DEVICES=4,5,6,7

ALL_MODELS=(
    "ehevolver" 
)

SCRIPT_ARGS=(
    "--gpus" "4,5,6,7"
    "--ddp"
    "--jump-horizon" "10"
    "--num-of-simulations" "100" 
    "--num-of-steps" "100"
    "--find-unused-parameters"
)

TORCHRUN_ARGS="--nproc_per_node=4"
SCRIPT_NAME="train_model.py" # Output updated to match your active file
MASTER_PORT=29504

# ================= Dataset List =================
DATASETS=(
    "plane_re40-80_fr1"
    "plane_re60_fr1-2"
    "point_re40-80_fr1"
    "point_re60_fr1-2"
)

# ================= Main Loop =================

echo "########################################################"
echo "Start Optimized Training: Loop by dataset, reuse data internally in Python"
echo "########################################################"

for data_name in "${DATASETS[@]}"; do
    echo ""
    echo "========================================================"
    echo "[$(date)] >>> Switching dataset: $data_name"
    echo "========================================================"
    
    data_path="dataset/${data_name}"
    
    # The run-name here is just a prefix for Swanlab/Log,
    # actually Python internally will overwrite it with {model_name}_{experiment_id}
    base_run_name="AllModels_${data_name}"

    # Concatenate model list string
    # Result like: "gno mppde meshgraphnet ..."
    MODELS_STRING="${ALL_MODELS[*]}"

    # Execute command
    # We pass all models to --model-list at once
    torchrun --master_port $MASTER_PORT $TORCHRUN_ARGS $SCRIPT_NAME \
        "${SCRIPT_ARGS[@]}" \
        --data-folder $data_path \
        --run-name $base_run_name \
        --model-list $MODELS_STRING

    if [ $? -ne 0 ]; then
        echo "!!! Error: Training sequence on dataset $data_name interrupted!"
        # exit 1 
    fi
    
    echo "[$(date)] All models training on dataset $data_name completed."
    
    # Rest for a while, clean up DDP processes
    sleep 20
done

echo "########################################################"
echo "All tasks completed"
echo "########################################################"