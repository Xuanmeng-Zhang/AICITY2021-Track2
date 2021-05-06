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
import argparse

def main():
    weight_dir = FLAGS.model_path
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    inference_program, feed_target_names, fetch_targets = fluid.io.load_inference_model(weight_dir, exe, 
                                                    model_filename='__model__', params_filename='__params__')

    img_list = glob(FLAGS.data_path + "*.jpg")

    result_dic = {}
    for info in tqdm(img_list):
        im_ori = cv2.imread(info)
        w = im_ori.shape[1]
        h = im_ori.shape[0]
        im_size = np.array([[im_ori.shape[0], im_ori.shape[1]]], dtype=np.int32)
        im = cv2.cvtColor(im_ori, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (224, 224))
        im = im.astype(np.float32, copy=False)
        im = im / 255.0
        im = im - np.array([0.485, 0.456, 0.406], dtype='float32')
        im = im / np.array([0.229, 0.224, 0.225], dtype='float32')
        #     print(im[0,0,2],im.shape)
        im = np.swapaxes(im, 1, 2)
        im = np.swapaxes(im, 1, 0)

        output = exe.run(inference_program, fetch_list= fetch_targets, 
                                     feed = {feed_target_names[0]:im[np.newaxis,:], feed_target_names[1]:im_size},
                                     return_numpy=False)

        result_dic[info] = {}
        result_dic[info]["top"] = []
        result_dic[info]["front"] = []
        result_dic[info]["left"] = []
        result_dic[info]["back"] = []
        result_dic[info]["right"] = []

        prob_th = 0.5
        for i in np.array(output[0]):
            if i[1] > prob_th:
                cls = i[0]
                x_min = max(int(i[2]),0)
                x_max = min(int(i[4]),w)
                y_min = max(int(i[3]),0)
                y_max = min(int(i[5]),h)
                prob = i[1]

                if int(cls) == 0:
                    type_name = "top"
                elif int(cls) == 1:
                    type_name = "front"
                elif int(cls) == 2:
                    type_name = "left"    
                elif int(cls) == 3:
                    type_name = "back"    
                elif int(cls) == 4:
                    type_name = "right"
                else:
                    print("error!")

                if len(result_dic[info][type_name]) == 0 or result_dic[info][type_name][0] < prob:
                    result_dic[info][type_name] = [prob,x_min,x_max,y_min,y_max]

    for i in tqdm(result_dic.keys()):
        data_set = i.split("/")[-2]
        img_name = i.split("/")[-1]
        img = cv2.imread(i)
        for part in result_dic[i].keys():
            if len(result_dic[i][part]) > 0:
                x1 = result_dic[i][part][1]
                y1 = result_dic[i][part][3]
                x2 = result_dic[i][part][2]
                y2 = result_dic[i][part][4]
                save_dir = "./aicity_crop_data/image_query/" + part
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = save_dir + "/" + img_name
                save_image = img[y1:y2,x1:x2,:]
                cv2.imwrite(save_path,save_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./dataset/aicity_data/image_query/",
        type=str,
        help="inference data dir")
    parser.add_argument(
        "--model_path",
        default='./inference_aicity_model/yolov3_r34/',
        type=str,
        help="inference model dir")
    FLAGS = parser.parse_args()
    main()