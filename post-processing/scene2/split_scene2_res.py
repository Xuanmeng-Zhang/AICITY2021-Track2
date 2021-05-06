import os
import pdb
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from combine_strategy_s2 import direction_strategy_21_s2


with open('real_query_cam.txt','r') as fid:
    query_cam_dict = {line.strip().split(' ')[0]:line.strip().split(' ')[1] for line in fid.readlines()}
with open('real_test_cam.txt','r') as fid:
    gallery_cam_dict = {line.strip().split(' ')[0]:line.strip().split(' ')[1] for line in fid.readlines()}

scene2_cam_list = ['c041', 'c042', 'c043', 'c044', 'c045','c046']

track_count = 0
gallery_track_dict = {}
gallery_track_dict_id = {}
with open('test_track.txt','r') as fid:
    lines = [line.strip() for line in fid.readlines()]
    for line in lines:
        track_imgs = line.split(' ')
        cam_set = set()
        for each in track_imgs:
            gallery_track_dict[each] = track_count
            gallery_track_dict_id[int(each[:-4])-1] = track_count
        track_count += 1

track_count = 0

print('tracklet info loaded')
with open('real_query_shape.txt','r') as fid:
    lines = [line.strip() for line in fid.readlines()]
    query_areas = []
    for each in lines:
        img_name, h, w = each.split(' ')
        area = int(h) * int(w)
        query_areas.append(area)

with open('real_test_shape.txt','r') as fid:
    lines = [line.strip() for line in fid.readlines()]
    gallery_areas = []
    for each in lines:
        img_name, h, w = each.split(' ')
        area = int(h) * int(w)
        gallery_areas.append(area)
    

with open('feas/real_query_fea_all_combine_pseudo.pkl','rb') as fid:
    query2 = pickle.load(fid)

with open('feas/real_gallery_fea_all_combine_pseudo.pkl','rb') as fid:
    gallery2 = pickle.load(fid)
query_direction_fea_dic = np.load("part_direction/query_direction_fea.npy",allow_pickle=True, encoding='latin1').item()
gallery_direction_fea_dic = np.load("part_direction/test_direction_fea.npy",allow_pickle=True, encoding='latin1').item()

query_part_cls_fea_dic = np.load("part_direction/query_part_cls_fea.npy",allow_pickle=True, encoding='latin1').item()
gallery_part_cls_fea_dic = np.load("part_direction/test_part_cls_fea.npy",allow_pickle=True, encoding='latin1').item()

with open('part_direction/backup/real_query_part_fea_aic21_combine_all_back.pkl','rb') as fid:
    query_back_fea_dict = pickle.load(fid)
with open('part_direction/backup/real_gallery_part_fea_aic21_combine_all_back.pkl','rb') as fid:
    gallery_back_fea_dict = pickle.load(fid)
with open('part_direction/backup/real_query_part_fea_aic21_combine_all_front.pkl','rb') as fid:
    query_front_fea_dict = pickle.load(fid)
with open('part_direction/backup/real_gallery_part_fea_aic21_combine_all_front.pkl','rb') as fid:
    gallery_front_fea_dict = pickle.load(fid)
with open('part_direction/backup/real_query_part_fea_aic21_combine_all_left_right.pkl','rb') as fid:
    query_left_right_fea_dict = pickle.load(fid)
with open('part_direction/backup/real_gallery_part_fea_aic21_combine_all_left_right.pkl','rb') as fid:
    gallery_left_right_fea_dict = pickle.load(fid)
with open('part_direction/backup/real_query_part_fea_aic21_combine_all_top.pkl','rb') as fid:
    query_top_fea_dict = pickle.load(fid)
with open('part_direction/backup/real_gallery_part_fea_aic21_combine_all_top.pkl','rb') as fid:
    gallery_top_fea_dict = pickle.load(fid)
    

with open('part_direction/backup/real_query_part_fea_aic21_combine_all_back_flip.pkl','rb') as fid:
    query_back_flip_fea_dict = pickle.load(fid)
with open('part_direction/backup/real_gallery_part_fea_aic21_combine_all_back_flip.pkl','rb') as fid:
    gallery_back_flip_fea_dict = pickle.load(fid)
with open('part_direction/backup/real_query_part_fea_aic21_combine_all_front_flip.pkl','rb') as fid:
    query_front_flip_fea_dict = pickle.load(fid)
with open('part_direction/backup/real_gallery_part_fea_aic21_combine_all_front_flip.pkl','rb') as fid:
    gallery_front_flip_fea_dict = pickle.load(fid)
with open('part_direction/backup/real_query_part_fea_aic21_combine_all_left_right_flip.pkl','rb') as fid:
    query_left_right_flip_fea_dict = pickle.load(fid)
with open('part_direction/backup/real_gallery_part_fea_aic21_combine_all_left_right_flip.pkl','rb') as fid:
    gallery_left_right_flip_fea_dict = pickle.load(fid)
with open('part_direction/backup/real_query_part_fea_aic21_combine_all_top_flip.pkl','rb') as fid:
    query_top_flip_fea_dict = pickle.load(fid)
with open('part_direction/backup/real_gallery_part_fea_aic21_combine_all_top_flip.pkl','rb') as fid:
    gallery_top_flip_fea_dict = pickle.load(fid)    
    

skip_dic_top_query = np.load("part_direction/skip_dic_query_add_area.npy",allow_pickle=True, encoding='latin1').item()
skip_dic_top_test = np.load("part_direction/skip_dic_test_add_area.npy",allow_pickle=True, encoding='latin1').item()

skip_dic_left_right_query = np.load("part_direction/skip_dic_left_right_query.npy",allow_pickle=True, encoding='latin1').item()
skip_dic_left_right_test = np.load("part_direction/skip_dic_left_right_test.npy",allow_pickle=True, encoding='latin1').item()

part_dims = 1536
top_dims = 1024

query_scene2_fea = []
query_scene2_id = []
query_scene2_cam = []
query_scene2_area = []
query_scene2_direction_fea = []
query_scene2_part_cls_fea = []
query_scene2_back_fea = []
query_scene2_front_fea = []
query_scene2_left_right_fea = []
query_scene2_back_flip_fea = []
query_scene2_front_flip_fea = []
query_scene2_left_right_flip_fea = []
query_scene2_top_fea = []
query_scene2_top_flip_fea = []


skip_scene2_mask_top = np.ones((540,14541))
skip_scene2_mask_left_right = np.ones((540,14541))

for i in range(1103):
    # print(i)
    query_name = str(i+1).zfill(6)+'.jpg'
    query_path =  query_name
    query_cam = query_cam_dict[query_name]
    
    if query_name in skip_dic_top_query.keys():
        query_top_ratio, area = skip_dic_top_query[query_name]
    else:
        query_top_ratio = 100
        area = 666
        
    if query_name in skip_dic_left_right_query.keys():
        query_left_right_ratio = skip_dic_left_right_query[query_name]
    else:
        query_left_right_ratio = 0

    direction_fea = query_direction_fea_dic[query_path]
    part_cls_fea = query_part_cls_fea_dic[query_path]
    # back
    if query_name in query_back_fea_dict.keys():
        back_fea = query_back_fea_dict[query_name]
        back_flip_fea = query_back_flip_fea_dict[query_name]
    else:
        back_fea = np.zeros(part_dims)
        back_flip_fea = np.zeros(part_dims)
    # front
    if query_name in query_front_fea_dict.keys():
        front_fea = query_front_fea_dict[query_name]
        front_flip_fea = query_front_flip_fea_dict[query_name]
    else:
        front_fea = np.zeros(part_dims)
        front_flip_fea = np.zeros(part_dims)
    # left right
    if query_name in query_left_right_fea_dict.keys():
        left_right_fea = query_left_right_fea_dict[query_name]
        left_right_flip_fea = query_left_right_flip_fea_dict[query_name]
    else:
        left_right_fea = np.zeros(part_dims)
        left_right_flip_fea = np.zeros(part_dims)
        
    if query_name in query_top_fea_dict.keys():
        top_fea = query_top_fea_dict[query_name]
        top_flip_fea = query_top_flip_fea_dict[query_name]
    else:
        top_fea = np.zeros(top_dims)
        top_flip_fea = np.zeros(top_dims)

    if query_cam in scene2_cam_list:
        if query_top_ratio < 0.2 or area < 400:
            skip_scene2_mask_top[len(query_scene2_fea),:] = 0
        if query_left_right_ratio > 2:
            skip_scene2_mask_left_right[len(query_scene2_fea),:] = 0
            
        query_scene2_fea.append(query2[query_name])
        query_scene2_id.append(i)
        query_scene2_cam.append(int(query_cam[1:]))
        query_scene2_area.append(query_areas[i])
        
        query_scene2_top_fea.append(top_fea)
        query_scene2_back_fea.append(back_fea)
        query_scene2_front_fea.append(front_fea)
        query_scene2_left_right_fea.append(left_right_fea)
        query_scene2_top_flip_fea.append(top_flip_fea)
        query_scene2_back_flip_fea.append(back_flip_fea)
        query_scene2_front_flip_fea.append(front_flip_fea)
        query_scene2_left_right_flip_fea.append(left_right_flip_fea)

        query_scene2_direction_fea.append(direction_fea)
        query_scene2_part_cls_fea.append(part_cls_fea)
       
        

query_scene2_fea = torch.FloatTensor(query_scene2_fea)
query_scene2_fea = F.normalize(query_scene2_fea)
query_scene2_top_fea = torch.FloatTensor(query_scene2_top_fea)
query_scene2_top_fea = F.normalize(query_scene2_top_fea)
query_scene2_back_fea = torch.FloatTensor(query_scene2_back_fea)
query_scene2_back_fea = F.normalize(query_scene2_back_fea)
query_scene2_front_fea = torch.FloatTensor(query_scene2_front_fea)
query_scene2_front_fea = F.normalize(query_scene2_front_fea)
query_scene2_left_right_fea = torch.FloatTensor(query_scene2_left_right_fea)
query_scene2_left_right_fea = F.normalize(query_scene2_left_right_fea)
query_scene2_top_flip_fea = torch.FloatTensor(query_scene2_top_flip_fea)
query_scene2_top_flip_fea = F.normalize(query_scene2_top_flip_fea)
query_scene2_back_flip_fea = torch.FloatTensor(query_scene2_back_flip_fea)
query_scene2_back_flip_fea = F.normalize(query_scene2_back_flip_fea)
query_scene2_front_flip_fea = torch.FloatTensor(query_scene2_front_flip_fea)
query_scene2_front_flip_fea = F.normalize(query_scene2_front_flip_fea)
query_scene2_left_right_flip_fea = torch.FloatTensor(query_scene2_left_right_flip_fea)
query_scene2_left_right_flip_fea = F.normalize(query_scene2_left_right_flip_fea)

query_scene2_direction_fea = torch.FloatTensor(query_scene2_direction_fea).squeeze(1)
#### test
query_scene2_direction_fea = F.softmax(query_scene2_direction_fea, dim=1)
###test
query_scene2_direction_fea = F.normalize(query_scene2_direction_fea)   
query_scene2_part_cls_fea = torch.FloatTensor(query_scene2_part_cls_fea).squeeze(1)
query_scene2_part_cls_fea = F.normalize(query_scene2_part_cls_fea)
query_scene2_id = np.array(query_scene2_id)
#pdb.set_trace()

print('query scene2 feature load done!')

gallery_scene2_fea = []
gallery_scene2_id = []
gallery_scene2_cam = []
gallery_scene2_track = []
gallery_scene2_area = []
gallery_scene2_direction_fea = []
gallery_scene2_part_cls_fea = []
gallery_scene2_back_fea = []
gallery_scene2_front_fea = []
gallery_scene2_left_right_fea = []
gallery_scene2_back_flip_fea = []
gallery_scene2_front_flip_fea = []
gallery_scene2_left_right_flip_fea = []
gallery_scene2_top_fea = []
gallery_scene2_top_flip_fea = []

for i in range(31238):
    gallery_name = str(i+1).zfill(6)+'.jpg'
    gallery_path = gallery_name
    gallery_cam = gallery_cam_dict[gallery_name]
    gallery_track = gallery_track_dict[gallery_name]

    if gallery_name in skip_dic_top_test.keys():
        gallery_top_ratio, area = skip_dic_top_test[gallery_name]
    else:
        gallery_top_ratio = 100
        area = 666
        
    if gallery_name in skip_dic_left_right_test.keys():
        gallery_left_right_ratio = skip_dic_left_right_test[gallery_name]
    else:
        gallery_left_right_ratio = 0
    
    direction_fea = gallery_direction_fea_dic[gallery_path]
    part_cls_fea = gallery_part_cls_fea_dic[gallery_path]
    # back
    if gallery_name in gallery_back_fea_dict.keys():
        back_fea = gallery_back_fea_dict[gallery_name]
        back_flip_fea = gallery_back_flip_fea_dict[gallery_name]
    else:
        #back_fea = np.zeros(512)
        back_fea = np.zeros(part_dims)
        back_flip_fea = np.zeros(part_dims)
    # front
    if gallery_name in gallery_front_fea_dict.keys():
        front_fea = gallery_front_fea_dict[gallery_name]
        front_flip_fea = gallery_front_flip_fea_dict[gallery_name]
    else:
        #front_fea = np.zeros(512)
        front_fea = np.zeros(part_dims)
        front_flip_fea = np.zeros(part_dims)
    # left right
    if gallery_name in gallery_left_right_fea_dict.keys():
        left_right_fea = gallery_left_right_fea_dict[gallery_name]
        left_right_flip_fea = gallery_left_right_flip_fea_dict[gallery_name]
    else:
        #left_right_fea = np.zeros(512)
        left_right_fea = np.zeros(part_dims)
        left_right_flip_fea = np.zeros(part_dims)
        
    if gallery_name in gallery_top_fea_dict.keys():
        top_fea = gallery_top_fea_dict[gallery_name]
        top_flip_fea = gallery_top_flip_fea_dict[gallery_name]
    else:
        #left_right_fea = np.zeros(512)
        top_fea = np.zeros(top_dims)
        top_flip_fea = np.zeros(top_dims)

    if gallery_cam in scene2_cam_list:
        
        if gallery_top_ratio < 0.2 or area < 400:
            skip_scene2_mask_top[:,len(gallery_scene2_fea)] = 0
        if gallery_left_right_ratio > 2:
            skip_scene2_mask_left_right[:,len(gallery_scene2_fea)] = 0
            
        gallery_scene2_fea.append(gallery2[gallery_name])
        gallery_scene2_id.append(i)
        gallery_scene2_cam.append(int(gallery_cam[1:]))
        gallery_scene2_track.append(gallery_track)
        gallery_scene2_area.append(gallery_areas[i])
           
        gallery_scene2_top_fea.append(top_fea)
        gallery_scene2_back_fea.append(back_fea)
        gallery_scene2_front_fea.append(front_fea)
        gallery_scene2_left_right_fea.append(left_right_fea)
        gallery_scene2_top_flip_fea.append(top_flip_fea)
        gallery_scene2_back_flip_fea.append(back_flip_fea)
        gallery_scene2_front_flip_fea.append(front_flip_fea)
        gallery_scene2_left_right_flip_fea.append(left_right_flip_fea)
        gallery_scene2_direction_fea.append(direction_fea)
        gallery_scene2_part_cls_fea.append(part_cls_fea)

gallery_scene2_fea = torch.FloatTensor(gallery_scene2_fea)
gallery_scene2_fea = F.normalize(gallery_scene2_fea)
gallery_scene2_top_fea = torch.FloatTensor(gallery_scene2_top_fea)
gallery_scene2_top_fea = F.normalize(gallery_scene2_top_fea)
gallery_scene2_back_fea = torch.FloatTensor(gallery_scene2_back_fea)
gallery_scene2_back_fea = F.normalize(gallery_scene2_back_fea)
gallery_scene2_front_fea = torch.FloatTensor(gallery_scene2_front_fea)
gallery_scene2_front_fea = F.normalize(gallery_scene2_front_fea)
gallery_scene2_left_right_fea = torch.FloatTensor(gallery_scene2_left_right_fea)
gallery_scene2_left_right_fea = F.normalize(gallery_scene2_left_right_fea)
gallery_scene2_top_flip_fea = torch.FloatTensor(gallery_scene2_top_flip_fea)
gallery_scene2_top_flip_fea = F.normalize(gallery_scene2_top_flip_fea)
gallery_scene2_back_flip_fea = torch.FloatTensor(gallery_scene2_back_flip_fea)
gallery_scene2_back_flip_fea = F.normalize(gallery_scene2_back_flip_fea)
gallery_scene2_front_flip_fea = torch.FloatTensor(gallery_scene2_front_flip_fea)
gallery_scene2_front_flip_fea = F.normalize(gallery_scene2_front_flip_fea)
gallery_scene2_left_right_flip_fea = torch.FloatTensor(gallery_scene2_left_right_flip_fea)
gallery_scene2_left_right_flip_fea = F.normalize(gallery_scene2_left_right_flip_fea)

gallery_scene2_direction_fea = torch.FloatTensor(gallery_scene2_direction_fea).squeeze(1)
### test
gallery_scene2_direction_fea = F.softmax(gallery_scene2_direction_fea, dim=1)
### test
gallery_scene2_direction_fea = F.normalize(gallery_scene2_direction_fea)
gallery_scene2_part_cls_fea = torch.FloatTensor(gallery_scene2_part_cls_fea).squeeze(1)
gallery_scene2_part_cls_fea = F.normalize(gallery_scene2_part_cls_fea)

gallery_scene2_id = np.array(gallery_scene2_id)

#skip_scene2_mask_top = np.ones((540,14541))
skip_scene2_mask_left_right = np.ones((540,14541))

print('gallery scene1 and scene2 feature load done!')
rank2 = direction_strategy_21_s2( query_scene2_direction_fea, gallery_scene2_direction_fea, 
                           query_scene2_part_cls_fea,gallery_scene2_part_cls_fea,
                           query_scene2_top_fea, gallery_scene2_top_fea,skip_scene2_mask_top,
                           query_scene2_back_fea,gallery_scene2_back_fea,
                           query_scene2_front_fea,gallery_scene2_front_fea,
                           query_scene2_left_right_fea,gallery_scene2_left_right_fea,skip_scene2_mask_left_right,
                           query_scene2_top_flip_fea, gallery_scene2_top_flip_fea,
                           query_scene2_back_flip_fea,gallery_scene2_back_flip_fea,
                           query_scene2_front_flip_fea,gallery_scene2_front_flip_fea,
                           query_scene2_left_right_flip_fea,gallery_scene2_left_right_flip_fea,
                           query_scene2_fea, gallery_scene2_fea, 
                           query_scene2_cam, gallery_scene2_cam, 
                           query_scene2_area, gallery_scene2_area, 
                           #gallery_scene2_track, query_scene2_id, gallery_scene2_id, replace_img_size=3200, part_factor=0.4)
                            gallery_scene2_track, 
                           query_scene2_id, gallery_scene2_id, replace_img_size=1500, part_factor=0.4)

with open('rank2.pkl','wb') as fid:
    pickle.dump(rank2, fid)
