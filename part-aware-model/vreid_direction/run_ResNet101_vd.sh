python -u train_direction_cos_decay.py --model_arch 'ResNet101_vd' --pretrain 'pretrained/ResNet101_vd_ssld_pretrained' --batch_size 32 --big_height 384 --big_width 384 --target_height 384 --target_width 384 --target_size 384 --learning_rate 0.01 --warm_up_iter 5000 --max_iter 45000 --lr_steps 25000 35000 40000  
 
