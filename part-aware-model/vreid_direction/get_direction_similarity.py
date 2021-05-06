from __future__ import division
import paddle
import paddle.fluid as fluid
import os
from tqdm import tqdm
import numpy as np
import pdb
import argparse
import time
import pickle
from skimage import exposure
from PIL import Image
import cv2
from glob import glob
import torch
import torch.nn.functional as F
import argparse

def main():

    weight_dir = FLAGS.model_path
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    inference_program, feed_target_names, fetch_targets = fluid.io.load_inference_model(weight_dir, exe, 
                                                    model_filename='model', params_filename='params')


    img_list = glob(FLAGS.data_path + "*.jpg")

    correct = 0
    result_dic = {}
    for info in tqdm(img_list):
        im_ori = cv2.imread(info)
        w = im_ori.shape[1]
        h = im_ori.shape[0]
        im_size = np.array([[im_ori.shape[0], im_ori.shape[1]]], dtype=np.int32)
        im = cv2.cvtColor(im_ori, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (384, 384))
        im = im.astype(np.float32, copy=False)
        im = im / 255.0
        im = im - np.array([0.485, 0.456, 0.406], dtype='float32')
        im = im / np.array([0.229, 0.224, 0.225], dtype='float32')
        im = np.swapaxes(im, 1, 2)
        im = np.swapaxes(im, 1, 0)

        output = exe.run(inference_program, fetch_list= fetch_targets, 
                                     feed = {feed_target_names[0]:im[np.newaxis,:]},
                                     return_numpy=True)
        #pred = np.argmax(output[0])
        result_dic[info] = output[0]
    
    np.save("./query_direction_res101_fea",result_dic)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./dataset/aicity_data/image_query/",
        type=str,
        help="inference data dir")
    parser.add_argument(
        "--model_path",
        default='./infer_model_direction/ResNet101_vd_sync/',
        type=str,
        help="inference model dir")
    FLAGS = parser.parse_args()
    main()
