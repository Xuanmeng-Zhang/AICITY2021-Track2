import os
import pdb
import numpy as np
import pickle

import argparse
import functools
def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('txt_name',  str,    'track2.txt',     "The txt file to check.")
args = parser.parse_args()

txt_name = args.txt_name


with open('txt_files/real_query_cam.txt','r') as fid:
    query_cam_dict = {line.strip().split(' ')[0]:line.strip().split(' ')[1] for line in fid.readlines()}
with open('txt_files/real_test_cam.txt','r') as fid:
    gallery_cam_dict = {line.strip().split(' ')[0]:line.strip().split(' ')[1] for line in fid.readlines()}

scene2_cam_list = ['c041', 'c042', 'c043', 'c044', 'c045','c046']

query_scene1_id = []
query_scene2_id = []
for i in range(1103):
    query_name = str(i+1).zfill(6)+'.jpg'
    query_cam = query_cam_dict[query_name]
    if query_cam in scene2_cam_list:
        query_scene2_id.append(i)
    else:
        query_scene1_id.append(i)


gallery_scene1_id = []
gallery_scene2_id = []
for i in range(31238):
    gallery_name = str(i+1).zfill(6)+'.jpg'
    gallery_cam = gallery_cam_dict[gallery_name]
    if gallery_cam in scene2_cam_list:
        gallery_scene2_id.append(i)
    else:
        gallery_scene1_id.append(i)
query_scene1_id = np.array(query_scene1_id)
query_scene2_id = np.array(query_scene2_id)
gallery_scene1_id = np.array(gallery_scene1_id)
gallery_scene2_id = np.array(gallery_scene2_id)



with open('scene1/rank1.pkl', 'rb') as fid:
    rank1 = pickle.load(fid)
with open('scene2/rank2.pkl', 'rb') as fid:
    rank2 = pickle.load(fid)
scene1_count = 0
scene2_count = 0
fid = open(txt_name,'w')
for i in range(1103):
    if i in query_scene1_id:
        cur_index = rank1[scene1_count]
        cur_index = gallery_scene1_id[cur_index]
        scene1_count += 1
    else:
        cur_index = rank2[scene2_count]
        cur_index = gallery_scene2_id[cur_index]
        scene2_count += 1
    cur_index = np.array(cur_index) + 1
    cur_index = cur_index.tolist()
    cur_index = list(map(str, cur_index))
    write_line = ' '.join(cur_index)
    fid.write(write_line + '\n')
fid.close()
