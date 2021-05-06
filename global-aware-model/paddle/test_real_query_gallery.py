from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import os
import time
import argparse
import functools
import numpy as np
import pickle

import paddle.fluid as fluid

#from reid.dataset.dataset import Dataset
from reid.data.source.dataset import Dataset
from reid.data import create_reader
from reid.model import model_creator



from config import cfg, parse_args, print_arguments, print_arguments_dict 



def main(cfg):
    ReidDataset = Dataset(root = cfg.test_data_dir)
    ReidDataset.load_query()
    ReidDataset.load_gallery()
    query_source = ReidDataset.query
    gallery_source = ReidDataset.gallery
    query_num = len(query_source)
    gallery_num = len(gallery_source)
    query_names = [ query_source[i][0] for i in range(query_num)]
    gallery_names = [ gallery_source[i][0] for i in range(gallery_num)]




    image = fluid.layers.data(name='image', shape=[None, 3, cfg.target_height, cfg.target_width], dtype='float32')
    #label = fluid.layers.data(name='label', shape=[None, 1], dtype='int64')
    index = fluid.layers.data(name='index', shape=[None, 1], dtype='int64')
    data_loader = fluid.io.DataLoader.from_generator(feed_list=[image, index], capacity=128, use_double_buffer=True, iterable=True)

    model = model_creator(cfg.model_arch)
    cls_out, fea_out = model.net(input=image, is_train=False, class_dim=1802, num_features = cfg.num_features)
    index = fluid.layers.cast(index, dtype='int32') 
    fea_out = fluid.layers.l2_normalize(x=fea_out, axis=1)

    pred_list = [fea_out, index]
    test_prog = fluid.default_main_program().clone(for_test=True)
    image.persistable = True
    fea_out.persistable = True
 
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())


    load_dir = os.path.join(cfg.model_save_dir, cfg.model_arch, cfg.weights)
    print(load_dir)
    def if_exist(var):
        if os.path.exists(os.path.join(load_dir, var.name)):
            print(var.name)
            return True
        else:
            return False
    fluid.io.load_vars(
        exe, load_dir, main_program=test_prog, predicate=if_exist)


    test_prog = fluid.CompiledProgram(test_prog).with_data_parallel()


    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))
    print("Found {} CUDA devices.".format(devices_num))
    if devices_num==1:
        places = fluid.cuda_places(0)
    else:
        places = fluid.cuda_places()
    reader_config = {'dataset':query_source, 
                     'img_dir': cfg.test_data_dir+'/image_query',
                     'batch_size':cfg.test_batch_size,
                     'num_instances':cfg.num_instances,
                     'is_test':True,
                     'sample_type':'Base',
                     'shuffle':False,
                     'drop_last':False,
                     'worker_num':8,
                     #'worker_num':-2,
                     'bufsize':32,
                     'input_fields':['image','index'],
                     'cfg':cfg}
    new_reader, _, _, _ = create_reader(reader_config)
    data_loader.set_sample_list_generator(new_reader, places = places)


    start_time = time.time()
    query_dict = {}
    count = 0

    for data in data_loader():
        out = exe.run(test_prog, fetch_list=[v.name for v in pred_list], feed = data, return_numpy=True)

        feas = out[0]
        cur_index = out[1].flatten().tolist()
        count = count+ len(cur_index)
        for single_index, fea in zip(cur_index, feas):
            fname = query_names[single_index]
            query_dict[fname] = fea

        cur_time = time.time() - start_time
        start_time = time.time()
        output_str = '{}/{}imgs, time:{} '.format(count, query_num, cur_time )
        print(output_str)
    if count == query_num:
        print('query features extract Done!!!')


    with open('real_query_fea_'+cfg.model_arch+'.pkl','wb') as fid:
        pickle.dump(query_dict,fid)




    reader_config = {'dataset':gallery_source, 
                     'img_dir': cfg.test_data_dir+'/image_test',
                     'batch_size':cfg.test_batch_size,
                     'num_instances':cfg.num_instances,
                     'is_test':True,
                     'sample_type':'Base',
                     'shuffle':False,
                     'drop_last':False,
                     'worker_num':8,
                     #'worker_num':-2,
                     'bufsize':32,
                     'input_fields':['image','index'],
                     'cfg':cfg}
    

    new_reader, _, _, _ = create_reader(reader_config)
    data_loader.set_sample_list_generator(new_reader, places = places)
    start_time = time.time()
    gallery_dict = {}
    count = 0

    for data in data_loader():
        out = exe.run(test_prog, fetch_list=[v.name for v in pred_list], feed = data, return_numpy=True)

        feas = out[0]
        cur_index = out[1].flatten().tolist()
        count = count+ len(cur_index)
        for single_index, fea in zip(cur_index, feas):
            fname = gallery_names[single_index]
            gallery_dict[fname] = fea

        cur_time = time.time() - start_time
        start_time = time.time()
        output_str = '{}/{}imgs, time:{} '.format(count, gallery_num, cur_time )
        print(output_str)
    if count == gallery_num:
        print('gallery features extract Done!!!')

    with open('real_gallery_fea_'+cfg.model_arch+'.pkl','wb') as fid:
        pickle.dump(gallery_dict,fid)






if __name__ == '__main__':
    args = parse_args()
    print_arguments_dict(args)
    main(args)
