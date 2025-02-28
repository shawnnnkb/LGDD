# TORCH_DISTRIBUTED_DEBUG=DETAIL
# tmux new -s train3d
# conda activate LGDD
unset LD_LIBRARY_PATH

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-19500} # if using multi-exp should change PORT
PORT=${PORT:-29500} # 
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

CUDA_VISIBLE_DEVICES="0,1,2,3" \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    --config $CONFIG \
    --launcher pytorch ${@:3}
    
# NOTE: remind train epochs in config file
# nohup python -u ./tools_det3d/train.py --config ./projects/RadarPillarNet/configs/TJ4D-radarpillarnet_4x1_80e.py > TJ4D-radarpillarnet_4x1_80e.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/RadarPillarNet/configs/TJ4D-radarpillarnet_4x4_20e.py 4 > TJ4D-radarpillarnet_4x4_20e.log 2>&1 &


