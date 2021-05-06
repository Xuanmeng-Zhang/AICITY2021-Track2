If you want to train the  pytorch models, you can run:

```bash
cd pytorch
# ResNet-101-ibn
python3 projects/AICITY21/train_net.py --config-file projects/AICITY21/configs/sbs_R101-ibn.yml  --num-gpus 2

# SeReNet-101-ibn
python3 projects/AICITY21/train_net.py --config-file projects/AICITY21/configs/sbs_SE101-ibn.yml  --num-gpus 2

# ReSNeSt-101
python3 projects/AICITY21/train_net.py --config-file projects/AICITY21/configs/sbs_S101-ibn.yml  --num-gpus 2

# Pseudo label model
python3 projects/AICITY21/train_net.py --config-file projects/AICITY21/configs/sbs_R101-ibn-pseudo-v1.yml  --num-gpus 2

python3 projects/AICITY21/train_net.py --config-file projects/AICITY21/configs/sbs_R101-ibn-pseudo-v2.yml  --num-gpus 2

# Part-aware model
python3 projects/AICITY21/train_net.py --config-file projects/AICITY21/configs/sbs_R101-ibn-back.yml --num-gpus 2

python3 projects/AICITY21/train_net.py --config-file projects/AICITY21/configs/sbs_R101-ibn-front.yml  --num-gpus 2

python3 projects/AICITY21/train_net.py --config-file projects/AICITY21/configs/sbs_R101-ibn-left-right.yml  --num-gpus 2

python3 projects/AICITY21/train_net.py --config-file projects/AICITY21/configs/sbs_R101-ibn-top.yml  --num-gpus 2

```



If you want to train the  paddlepaddle models, you can run:

```sh
cd paddle

# HRNet
python -u train_pid_color_type_all_cos_decay.py --model_arch 'HRNet_W48_C' --pretrain 'pretrained/HRNet_W48_C_ssld_pretrained' --batch_size 32 --big_height 384 --big_width 384 --target_height 384 --target_width 384 --learning_rate 0.01 --warm_up_iter 5000 --max_iter 45000 --lr_steps 25000 35000 40000

python -u train_pid_color_type_all_cos_decay.py --model_arch 'HRNet_W48_C' --pretrain 'pretrained/HRNet_W48_C_ssld_pretrained' --batch_size 32 --big_height 400 --big_width 400 --target_height 384 --target_width 384 --learning_rate 0.01 --warm_up_iter 5000 --max_iter 45000 --lr_steps 25000 35000 40000

python -u train_pid_color_type_all_cos_decay.py --model_arch 'HRNet_W48_C' --pretrain 'pretrained/HRNet_W48_C_ssld_pretrained' --batch_size 32 --big_height 416 --big_width 416 --target_height 416 --target_width 416 --learning_rate 0.01 --warm_up_iter 5000 --max_iter 45000 --lr_steps 25000 35000 40000

python -u train_pid_color_type_all_cos_decay.py --model_arch 'HRNet_W48_C' --pretrain 'pretrained/HRNet_W48_C_ssld_pretrained' --batch_size 32 --big_height 432 --big_width 432 --target_height 416 --target_width 416 --learning_rate 0.01 --warm_up_iter 5000 --max_iter 45000 --lr_steps 25000 35000 40000

# Res2net200
python -u train_pid_color_type_all_cos_decay.py --model_arch 'Res2Net200_vd' --pretrain 'pretrained/Res2Net200_vd_26w_4s_ssld_pretrained' --batch_size 32 --big_height 400 --big_width 400 --target_height 384 --target_width 384 --learning_rate 0.012 --warm_up_iter 5000 --max_iter 45000 --lr_steps 25000 35000 40000 

python -u train_pid_color_type_all_cos_decay.py --model_arch 'Res2Net200_vd' --pretrain 'pretrained/Res2Net200_vd_26w_4s_ssld_pretrained' --batch_size 32 --big_height 384 --big_width 384 --target_height 384 --target_width 384 --learning_rate 0.003 --warm_up_iter 5000 --max_iter 45000 --lr_steps 25000 35000 40000  

# Part-aware model
python -u train_back.py --model_arch 'HRNet_W48_C' --pretrain 'pretrained/HRNet_W48_C_ssld_pretrained' --batch_size 32 --big_height 400 --big_width 400 --target_height 384 --target_width 384 --learning_rate 0.01 --warm_up_iter 5000 --max_iter 45000 --lr_steps 25000 35000 40000

python -u train_left_right.py --model_arch 'HRNet_W48_C' --pretrain 'pretrained/HRNet_W48_C_ssld_pretrained' --batch_size 32 --big_height 400 --big_width 400 --target_height 384 --target_width 384 --learning_rate 0.01 --warm_up_iter 5000 --max_iter 45000 --lr_steps 25000 35000 40000

python -u train_front.py --model_arch 'HRNet_W48_C' --pretrain 'pretrained/HRNet_W48_C_ssld_pretrained' --batch_size 32 --big_height 400 --big_width 400 --target_height 384 --target_width 384 --learning_rate 0.01 --warm_up_iter 5000 --max_iter 45000 --lr_steps 25000 35000 40000

python -u train_top.py --model_arch 'HRNet_W48_C' --pretrain 'pretrained/HRNet_W48_C_ssld_pretrained' --batch_size 32 --big_height 400 --big_width 400 --target_height 384 --target_width 384 --learning_rate 0.01 --warm_up_iter 5000 --max_iter 45000 --lr_steps 25000 35000 40000

```

