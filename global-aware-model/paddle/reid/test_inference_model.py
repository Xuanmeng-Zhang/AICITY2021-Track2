from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import os

import argparse
import functools
import numpy as np
from reader import Data_generator,Train_Dataloader
from config import cfg, parse_args, print_arguments 
from model import model_creator
from model.resnet_reid import ResNet18
import paddle.fluid as fluid
from learning_rate import exponential_with_warmup_decay
import time
import pickle

def main(cfg):
    num_box = 1
    img_mean = np.array([123.675, 116.28, 103.53],dtype=np.float32)
    img_std = np.array([58.395, 57.12 , 57.375],dtype=np.float32)

    data_gen = Data_generator(root_dir=cfg.data_dir, 
                                    batch_size=cfg.test_batch_size, drop_last = True, 
                                    split_id=0, num_instances = 1, 
                                    big_height= cfg.big_height, big_width=cfg.big_width, 
                                    target_height=cfg.target_height, target_width=cfg.target_width)

    #query_data_gen = data_gen.query_generator_infer_model()
    query_data_gen = data_gen.query_generator()
    #gallery_data_gen = data_gen.gallery_generator_infer_model()

    query_num = len(data_gen.query_source)
    gallery_num = len(data_gen.gallery_source)
    train_class_num = data_gen.trainval_pid_nums
 
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    load_dir = os.path.join(cfg.model_save_dir, cfg.model_arch, 'infer_model')
    inference_program, feed_target_names, fetch_targets = fluid.io.load_inference_model(load_dir, exe, model_filename='model', params_filename='params')


    start_time = time.time()
    query_dict = {}
    count = 0
    total_time = 0.0
    for batch_data in query_data_gen:
        for j in range(len(batch_data)):
            count +=1
            img, fname, pid, camid = batch_data[j]
            fea = exe.run(inference_program, fetch_list= fetch_targets, feed = { feed_target_names[0]:img[np.newaxis,:]},return_numpy=True)
            query_dict[fname] = fea[0][0]
            #pdb.set_trace()

            cur_time = time.time() - start_time
            start_time = time.time()
            total_time +=cur_time

            output_str = '{}/{}imgs, time:{} '.format(count, query_num, cur_time )
            print(output_str)
            #pdb.set_trace()


    with open('query_fea_inference_model.pkl','w') as fid:
        pickle.dump(query_dict,fid)
    print('average_time:{}'.format(total_time/count))

# 

if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    main(args)
