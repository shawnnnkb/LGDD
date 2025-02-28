CONFIG_PATH=./projects/LGDD/configs/VoD-LGDD_4x4_24e.py
CHECKPOINT_PATH=./projects/LGDD/checkpoints/VoD-best.pth
OUTPUT_NAME=vod-LGDD
PRED_RESULTS=./tools_det3d/view-of-delft-dataset/pred_results/$OUTPUT_NAME 

# CONFIG_PATH=./projects/RadarPillarNet/configs/VoD-radarpillarnet_4x1_80e.py
# CHECKPOINT_PATH=./projects/RadarPillarNet/checkpoints/VoD-baseline.pth
# OUTPUT_NAME=vod-RadarPillarNet
# PRED_RESULTS=./tools_det3d/view-of-delft-dataset/pred_results/$OUTPUT_NAME

GPUS="4"
PORT=${PORT:-29500}
CUDA_VISIBLE_DEVICES="0,1,3,4" \
# CUDA_VISIBLE_DEVICES="0" \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/tools_det3d/test.py \
    --format-only \
    --eval-options submission_prefix=$PRED_RESULTS \
    --config $CONFIG_PATH \
    --checkpoint $CHECKPOINT_PATH \
    --launcher pytorch ${@:4}

# python tools_det3d/test.py \
# --format-only \
# --eval-options submission_prefix=$PRED_RESULTS \
# --config $CONFIG_PATH \
# --checkpoint $CHECKPOINT_PATH

python tools_det3d/view-of-delft-dataset/FINAL_EVAL.py \
--pred_results $PRED_RESULTS
