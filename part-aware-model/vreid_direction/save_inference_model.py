from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import os

import argparse
import functools
import numpy as np
# from reader import Data_generator,Train_Dataloader
from config import cfg, parse_args, print_arguments 
from reid.model import model_creator
# from model.resnet_reid import ResNet18
import paddle.fluid as fluid
# from learning_rate import exponential_with_warmup_decay
import time
import pickle

## 1 x = x/255 2 (x-mean)/std = 

# img_mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
img_mean = np.array([123.675, 116.28, 103.53])
# img_std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
img_std = np.array([58.395, 57.12 , 57.375])


def main(cfg):
#     data_gen = Data_generator(root_dir=cfg.data_dir, 
#                                     batch_size=cfg.test_batch_size, drop_last = True, 
#                                     split_id=0, num_instances = 1, 
#                                     big_height= cfg.big_height, big_width=cfg.big_width, 
#                                     target_height=cfg.target_height, target_width=cfg.target_width)

    train_class_num = 8


    image_shape = [3, cfg.target_height,cfg.target_width]
    image_data = fluid.layers.data(name='image_data', shape=[-1]+image_shape, dtype='float32')

    model = model_creator(cfg.model_arch)
    cls_out, fea_out = model.net_orientation(input=image_data, is_train=False, class_dim=train_class_num, num_features = cfg.num_features)

    pred_list = [cls_out]
    test_prog = fluid.default_main_program().clone(for_test=True)
    image_data.persistable = True
    fea_out.persistable = True
 
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program()) 

    #feeder = fluid.DataFeeder(place=place, feed_list = [image_data, image_mean, image_std])

    
    load_weight_dir = os.path.join(cfg.model_save_dir, cfg.model_arch, cfg.weights)
    print(load_weight_dir)
    def if_exist(var):
        if os.path.exists(os.path.join(load_weight_dir, var.name)):
            print(var.name)
            return True
        else:
            return False
    fluid.io.load_vars(
        exe, load_weight_dir, main_program=test_prog, predicate=if_exist)
    save_dir = os.path.join(cfg.model_save_dir, cfg.model_arch, 'infer_model')
    fluid.io.save_inference_model(save_dir, ['image_data'], pred_list, exe, main_program=test_prog, model_filename='model', params_filename='params')
    print('save inference model Done!')

#


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    main(args)
