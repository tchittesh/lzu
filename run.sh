#!/bin/bash

outDir="/project_data/ramanan/cthavama/LZU_release_tests"  # CHANGE THIS
nGPU=2

case $1 in
    "fcos3d_0.25"|"fcos3d_0.50"|"fcos3d_0.75"|"fcos3d_1.00"|"lzu_fcos3d_0.25"|"lzu_fcos3d_0.50"|"lzu_fcos3d_0.75"|"lzu_fcos3d_1.00") expName=$1 ;;
    *) echo "Invalid experiment name." && exit;;
esac

# Test checkpointed model
# set data.test.samples_per_gpu=1 if runnning timing tests
# python tools/test.py \
#     configs/$expName.py \
#     ckpt/$expName.pth \
#     --out $outDir/$expName/test_checkpoint/results.pkl \
#     --eval bbox \
#     --cfg-options \
#         data.test.samples_per_gpu=8 \

# Train
torchrun \
    --nnodes 1 \
    --nproc_per_node $nGPU \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0 \
    tools/train.py \
    configs/$expName.py \
    --work-dir $outDir/$expName/work \
    --gpus $nGPU \
    --launcher pytorch \

# Test trained model
# set data.test.samples_per_gpu=1 if runnning timing tests
# python tools/test.py \
#     configs/$expName.py \
#     $outDir/$expName/work/latest.pth \
#     --out $outDir/$expName/test/results.pkl \
#     --eval bbox \
#     --cfg-options \
#         data.test.samples_per_gpu=8 \
