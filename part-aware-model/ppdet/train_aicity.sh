export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_IB_GID_INDEX=3
python -u tools/train.py -c configs/yolov3_r34.yml --eval

