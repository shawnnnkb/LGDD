CONFIG_PATH=./projects/LGDD/configs/TJ4D-LGDD_4x4_24e.py
CHECKPOINT_PATH=./projects/LGDD/checkpoints/TJ4D-best.pth

# CONFIG_PATH=./projects/RadarPillarNet/configs/TJ4D-radarpillarnet_4x4_20e.py
# CHECKPOINT_PATH=./projects/RadarPillarNet/checkpoints/TJ4D-baseline.pth

GPUS="4"
PORT=${PORT:-29500}
CUDA_VISIBLE_DEVICES="0,1,3,4" \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/tools_det3d/test.py \
    --config $CONFIG_PATH \
    --checkpoint $CHECKPOINT_PATH \
    --eval mAP \
    --launcher pytorch ${@:4}

# python tools_det3d/test.py \
# --config  $CONFIG_PATH \
# --checkpoint $CHECKPOINT_PATH \
# --eval mAP