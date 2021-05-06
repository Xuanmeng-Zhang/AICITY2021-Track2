If you want to extract the pytorch features, you can run:

```bash
cd pytorch
# ResNet-101-ibn
python3 projects/AICITY21/train_net.py --config-file projects/AICITY21/configs/sbs_R101-ibn.yml  --eval-only MODEL.WEIGHTS logs/aicity21/sbs_R101-ibn/model_0011.pth TEST.IMS_PER_BATCH 256  MODEL.DEVICE "cuda:0" TEST.FLIP_ENABLED True

# SeReNet-101-ibn
python3 projects/AICITY21/train_net.py --config-file projects/AICITY21/configs/sbs_SE101-ibn.yml  --eval-only MODEL.WEIGHTS logs/aicity21/sbs_SE101-ibn/model_0011.pth TEST.IMS_PER_BATCH 256  MODEL.DEVICE "cuda:0" TEST.FLIP_ENABLED True

# ReSNeSt-101
python3 projects/AICITY21/train_net.py --config-file projects/AICITY21/configs/sbs_S101-ibn.yml  --eval-only MODEL.WEIGHTS logs/aicity21/sbs_S101/model_0011.pth TEST.IMS_PER_BATCH 256  MODEL.DEVICE "cuda:0" TEST.FLIP_ENABLED True

# Pseudo label model
python3 projects/AICITY21/train_net.py --config-file projects/AICITY21/configs/sbs_R101-ibn-pseudo-v1.yml  --eval-only MODEL.WEIGHTS logs/aicity21/sbs_R101-ibn-pseudo-v1/model_0011.pth TEST.IMS_PER_BATCH 256  MODEL.DEVICE "cuda:0" TEST.FLIP_ENABLED True

python3 projects/AICITY21/train_net.py --config-file projects/AICITY21/configs/sbs_R101-ibn-pseudo-v2.yml  --eval-only MODEL.WEIGHTS logs/aicity21/sbs_R101-ibn-pseudo-v2/model_0011.pth TEST.IMS_PER_BATCH 256  MODEL.DEVICE "cuda:0" TEST.FLIP_ENABLED True

# Part-aware model
python3 projects/AICITY21/train_net.py --config-file projects/AICITY21/configs/sbs_R101-ibn-back.yml  --eval-only MODEL.WEIGHTS logs/aicity21/sbs_R101-ibn-back/model_0011.pth TEST.IMS_PER_BATCH 256  MODEL.DEVICE "cuda:0"

python3 projects/AICITY21/train_net.py --config-file projects/AICITY21/configs/sbs_R101-ibn-front.yml  --eval-only MODEL.WEIGHTS logs/aicity21/sbs_R101-ibn-front/model_0011.pth TEST.IMS_PER_BATCH 256  MODEL.DEVICE "cuda:0"

python3 projects/AICITY21/train_net.py --config-file projects/AICITY21/configs/sbs_R101-ibn-left-right.yml --eval-only MODEL.WEIGHTS logs/aicity21/sbs_R101-ibn-left-right/model_0011.pth TEST.IMS_PER_BATCH 256  MODEL.DEVICE "cuda:0"

python3 projects/AICITY21/train_net.py --config-file projects/AICITY21/configs/sbs_R101-ibn-top.yml --eval-only MODEL.WEIGHTS logs/aicity21/sbs_R101-ibn-top/model_0011.pth TEST.IMS_PER_BATCH 256  MODEL.DEVICE "cuda:0"

```



If you want to extract the  paddlepaddle featuress, you can run:

```sh
cd paddle
# HRNet
python -u test.py --model_arch 'HRNet_W48_C' --weights model_final --big_height 432 --big_width 432 --target_height 416 --target_width 416

python -u test.py --model_arch 'HRNet_W48_C' --weights model_final --big_height 400 --big_width 400 --target_height 384 --target_width 384 

python -u test.py --model_arch 'HRNet_W48_C' --weights model_final --big_height 416 --big_width 416 --target_height 416 --target_width 416 

python -u test.py --model_arch 'HRNet_W48_C' --weights model_final --big_height 432 --big_width 432 --target_height 416 --target_width 416

# Res2net200
python -u test.py --model_arch --model_arch 'Res2Net200_vd' --weights model_final  --big_height 400 --big_width 400 --target_height 384 --target_width 384

python -u test.py --model_arch --model_arch 'Res2Net200_vd' --weights model_final  --big_height 384 --big_width 384 --target_height 384 --target_width 384

# Part-aware model
python -u test_real_query_gallery.py  --model_arch 'HRNet_W48_C' --weights model_final --big_height 432 --big_width 432 --target_height 416 --target_width 416

python -u test_real_query_gallery_flip.py  --model_arch 'HRNet_W48_C' --weights model_final --big_height 432 --big_width 432 --target_height 416 --target_width 416

```

