import os
import pdb
import numpy as np
import torch
import torch.nn.functional as F
import scipy.io
import math
from sklearn.cluster import DBSCAN
from re_ranking import re_ranking

def direction_strategy_21(query_direction_feas, gallery_direction_feas,
                          query_part_cls_feas, gallery_part_cls_feas,
                          query_top_feas, gallery_top_feas,skip_mask_top,
                          query_back_feas, gallery_back_feas,
                          query_front_feas, gallery_front_feas,
                          query_left_right_feas, gallery_left_right_feas,skip_mask_left_right,
                          query_top_flip_feas, gallery_top_flip_feas,
                          query_back_flip_feas, gallery_back_flip_feas,
                          query_front_flip_feas, gallery_front_flip_feas,
                          query_left_right_flip_feas, gallery_left_right_flip_feas,
                          query_feas, gallery_feas, query_cams, gallery_cams, 
                          query_areas, gallery_areas, 
                          gallery_tracklets, 
                          query_ids, gallery_ids, replace_img_size=15000, part_factor=0.3):

    num_query = query_feas.shape[0]
    num_gallery = gallery_feas.shape[0]

    query_feas = query_feas
    gallery_feas = gallery_feas
    
    query_cams = np.array(query_cams)
    gallery_cams = np.array(gallery_cams)
    query_areas = np.array(query_areas)
    gallery_areas = np.array(gallery_areas)
    gallery_tracklets = np.array(gallery_tracklets)
    query_ids = np.array(query_ids)
    gallery_ids = np.array(gallery_ids)
    unique_gallery_tracklets = np.unique(gallery_tracklets)
    num_gallery_class = len(unique_gallery_tracklets)


    def tracklets_mean_strategy(score_mat):
        for each in unique_gallery_tracklets:
            track_list = np.argwhere(gallery_tracklets == each).flatten()
            #track_list = gallery_tracklets[i]
            track_mat = score_mat[:,track_list]
            track_mat = np.sort(track_mat,axis=1)
            track_mat = track_mat[:,-1:]

            score_mat[:,track_list] = track_mat.reshape(num_query,1)
            #pdb.set_trace()
        return score_mat
    
    gallery_top_feas = F.normalize(gallery_top_feas, dim=1)
    query_top_feas = F.normalize(query_top_feas, dim=1)
    score_mat_top = torch.mm(query_top_feas, gallery_top_feas.transpose(0,1)).numpy()
    score_mat_top = score_mat_top * skip_mask_top
  
    gallery_top_flip_feas = F.normalize(gallery_top_flip_feas, dim=1)
    query_top_flip_feas = F.normalize(query_top_flip_feas, dim=1)
    score_mat_top_flip = torch.mm(query_top_flip_feas, gallery_top_flip_feas.transpose(0,1)).numpy()
    score_mat_top_flip = score_mat_top_flip * skip_mask_top
  
    score_mat_top[score_mat_top == 0] = -1
    score_mat_top_flip[score_mat_top_flip == 0] = -1
    
    gallery_back_feas = F.normalize(gallery_back_feas, dim=1)
    query_back_feas = F.normalize(query_back_feas, dim=1)
    score_mat_back = torch.mm(query_back_feas, gallery_back_feas.transpose(0,1)).numpy()
    gallery_back_flip_feas = F.normalize(gallery_back_flip_feas, dim=1)
    query_back_flip_feas = F.normalize(query_back_flip_feas, dim=1)
    score_mat_back_flip = torch.mm(query_back_flip_feas, gallery_back_flip_feas.transpose(0,1)).numpy()

    score_mat_back[score_mat_back == 0] = -1
    score_mat_back_flip[score_mat_back_flip == 0] = -1
    #np.save("./score_mat_back_check_1",score_mat_back)
    
    gallery_front_feas = F.normalize(gallery_front_feas, dim=1)
    query_front_feas = F.normalize(query_front_feas, dim=1)
    score_mat_front = torch.mm(query_front_feas, gallery_front_feas.transpose(0,1)).numpy()
    gallery_front_flip_feas = F.normalize(gallery_front_flip_feas, dim=1)
    query_front_flip_feas = F.normalize(query_front_flip_feas, dim=1)
    score_mat_front_flip = torch.mm(query_front_flip_feas, gallery_front_flip_feas.transpose(0,1)).numpy()

    score_mat_front[score_mat_front == 0] = -1
    score_mat_front_flip[score_mat_front_flip == 0] = -1
    #np.save("./score_mat_front_check_1",score_mat_front)
    
    gallery_left_right_feas = F.normalize(gallery_left_right_feas, dim=1)
    query_left_right_feas = F.normalize(query_left_right_feas, dim=1)
    score_mat_left_right = torch.mm(query_left_right_feas, gallery_left_right_feas.transpose(0,1)).numpy()
    score_mat_left_right = score_mat_left_right * skip_mask_left_right
    
    gallery_left_right_flip_feas = F.normalize(gallery_left_right_flip_feas, dim=1)
    query_left_right_flip_feas = F.normalize(query_left_right_flip_feas, dim=1)
    score_mat_left_right_flip = torch.mm(query_left_right_flip_feas, gallery_left_right_flip_feas.transpose(0,1)).numpy()
    score_mat_left_right_flip = score_mat_left_right_flip * skip_mask_left_right

    score_mat_left_right[score_mat_left_right == 0] = -1
    score_mat_left_right_flip[score_mat_left_right_flip == 0] = -1
    #np.save("./score_mat_left_right_check_1",score_mat_left_right)
    
    score_mat_top = tracklets_mean_strategy(score_mat_top)
    score_mat_back = tracklets_mean_strategy(score_mat_back)
    score_mat_front = tracklets_mean_strategy(score_mat_front)
    score_mat_left_right = tracklets_mean_strategy(score_mat_left_right)
    
    score_mat_top_flip = tracklets_mean_strategy(score_mat_top_flip)
    score_mat_back_flip = tracklets_mean_strategy(score_mat_back_flip)
    score_mat_front_flip = tracklets_mean_strategy(score_mat_front_flip)
    score_mat_left_right_flip = tracklets_mean_strategy(score_mat_left_right_flip)
    
    #score_mat_part_tmp = np.zeros((score_mat_back.shape[0],score_mat_back.shape[1],8))
    score_mat_part_tmp = np.zeros((score_mat_back.shape[0],score_mat_back.shape[1],6))
    score_mat_part_tmp[:,:,0] = score_mat_back
    score_mat_part_tmp[:,:,1] = score_mat_front
    score_mat_part_tmp[:,:,2] = score_mat_left_right
    score_mat_part_tmp[:,:,3] = score_mat_back_flip
    score_mat_part_tmp[:,:,4] = score_mat_front_flip
    score_mat_part_tmp[:,:,5] = score_mat_left_right_flip
    #score_mat_part_tmp[:,:,6] = score_mat_top
    #score_mat_part_tmp[:,:,7] = score_mat_top_flip
    
    score_mat_part = np.max(score_mat_part_tmp,axis=2)
    
    score_mat_part = torch.tensor(score_mat_part)
    score_mat_top = torch.tensor(score_mat_top)
    score_mat_back = torch.tensor(score_mat_back)
    score_mat_front = torch.tensor(score_mat_front)
    score_mat_left_right = torch.tensor(score_mat_left_right)
    
    #score_mask_part = (score_mat_back == -1) * (score_mat_front == -1) * (score_mat_left_right == -1) * (score_mat_top == -1)
    score_mask_part = (score_mat_back == -1) * (score_mat_front == -1) * (score_mat_left_right == -1)
    #score_mask_part = torch.tensor(score_mask_part)
    
    
    query_part_cls_feas = F.normalize(query_part_cls_feas, dim=1)
    gallery_part_cls_feas = F.normalize(gallery_part_cls_feas, dim=1)
    score_part_cls_total = torch.mm(query_part_cls_feas, gallery_part_cls_feas.transpose(0,1))

    query_direction_feas = F.normalize(query_direction_feas, dim=1)
    gallery_direction_feas = F.normalize(gallery_direction_feas, dim=1)
    score_direction_total = torch.mm(query_direction_feas, gallery_direction_feas.transpose(0,1))

    gallery_feas = F.normalize(gallery_feas, dim=1)
    query_feas = F.normalize(query_feas, dim=1)
    
    score_direction_total = score_direction_total.numpy()
    score_direction_total = tracklets_mean_strategy(score_direction_total)
    score_direction_total = torch.tensor(score_direction_total)
    
    
    score_part_cls_total = score_part_cls_total.numpy()
    score_part_cls_total = tracklets_mean_strategy(score_part_cls_total)
    score_part_cls_total = torch.tensor(score_part_cls_total)


    q_q_sim = torch.mm(query_feas, torch.transpose(query_feas, 0, 1))
##############################################################################################
    ## query query cluster
    q_q_dist = q_q_sim.cpu().numpy() 
    q_q_dist[q_q_dist>1] = 1  #due to the epsilon 
    q_q_dist = 2 - 2 * q_q_dist # [0,4]

    eps = 0.6
    min_samples= 2
    cluster1 = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', algorithm='auto', n_jobs=-1)
    cluster1 = cluster1.fit(q_q_dist)
    qlabels = cluster1.labels_
    ## pdb.set_trace()
    num_query_class = len(np.unique(cluster1.labels_))

    print('num_true_label ',np.where(cluster1.labels_!=-1)[0].shape)
    print('num_query_class ', num_query_class)
    unique_query_tracklets = np.unique(qlabels)
    query_tracklets  = qlabels

    unique_gallery_tracklets = np.unique(gallery_tracklets)
    num_gallery_class = len(unique_gallery_tracklets)
    print('query and gallery cluster done!')

####### use average feature to replace normal feature
    alpha = 1.0
    beta = 1.0
    query_feature_clone = query_feas.clone()
    junk_index_q = np.argwhere(query_areas < replace_img_size).flatten()
    if len(junk_index_q) > 0:
        for each in unique_query_tracklets:
            if each == -1: continue
            index = np.argwhere(query_tracklets == each).flatten()
            high_quality_index = np.setdiff1d(index, junk_index_q)
            if len(high_quality_index) == 0:
                high_quality_index = index
            qf_mean = torch.mean(query_feature_clone[high_quality_index, :], dim=0)
            for j in range(len(index)):
                #query_feas[index[j], :] = qf_mean
                query_feas[index[j],:] += alpha*qf_mean

##################################################################################################
    # use high quality img mean feature adding to low quality img feature
    gallery_feature_clone = gallery_feas.clone()
    junk_index_g = np.argwhere(gallery_areas < replace_img_size).flatten() # 150x150
    ####### whether use each tracklet expand need to check ###########################3
    if len(junk_index_g) > 0:
        for each in unique_gallery_tracklets:
            index = np.argwhere(gallery_tracklets == each).flatten()
            high_quality_index = np.setdiff1d(index, junk_index_g)
            if len(high_quality_index) == 0:
                high_quality_index = index
            gf_mean = torch.mean(gallery_feature_clone[high_quality_index,:], dim=0)
            for j in range(len(index)):
              ################# different #######################
                gallery_feas[index[j], :] += beta * gf_mean
                #gallery_feas[index[j], :] = 0.2 * gallery_feas[index[j], :] + 0.8 * gf_mean
    
    gallery_feas = F.normalize(gallery_feas, dim=1)
    query_feas = F.normalize(query_feas, dim=1)
    score_total = torch.mm(query_feas, gallery_feas.transpose(0,1))


    ### cam filter
    
    for i in range(num_query):
        q_cam = query_cams[i]
        ignore_index = np.argwhere(gallery_cams==q_cam).flatten()
        score_total[i,ignore_index] = score_total[i,ignore_index] - 1.0
    print('query and gallery same cam filter done!')

        
    #score_total = score_total - part_factor*((score_direction_total+score_part_cls_total+2)/4)*(1.0-torch.logical_not(score_mask_part)*score_mat_part-score_mask_part*score_total)
    score_total = score_total - part_factor*((score_direction_total+score_part_cls_total+2)/4)*(0.9-torch.logical_not(score_mask_part)*score_mat_part-score_mask_part*score_total)

    score_total = score_total.cpu().numpy()

    ntop=99 # delete the different tracklet with the same camera of top 5
    beta = 0.15
    for count in range(1):
        for j in range(ntop):
            for i in range(num_query):
                topk_index = np.argsort(score_total[i,:])
                topk_index = topk_index[-1-j]
                good_index = np.argwhere(gallery_tracklets==gallery_tracklets[topk_index]).flatten()
                bad_index = np.argwhere(gallery_cams==gallery_cams[topk_index]).flatten() # cam id is the same as the topk, but is not the same track
                ignore_index = np.setdiff1d(bad_index, good_index)
                score_total[i,ignore_index] = score_total[i,ignore_index] -beta/(1+j)
        print('punish sample which has same gallery cam with high similar topk samples done!')


    #use_k_reciprocal_re_rank = False
    use_k_reciprocal_re_rank = True
    #use_top_k_punish_again = False
    use_top_k_punish_again = True
    if not use_k_reciprocal_re_rank:
        score_total = torch.FloatTensor(score_total)
        score_total = score_total.cuda()
        
        _, final_rank = score_total.topk(k=100, dim=-1, largest=True, sorted=True)
        all_index = final_rank.data.cpu().numpy()
        print('no rerank ...')
        return all_index
    else:
    ################################### begin rerank $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        q_q_sim = torch.mm(query_feas, query_feas.t())
        g_g_sim = torch.mm(gallery_feas, gallery_feas.t())

        q_q_sim = q_q_sim.data.cpu().numpy()
        ### good_query-tracklet-sim to 1

        for i in range(num_query):
            if query_tracklets[i]==-1: continue
            good_index = np.argwhere(query_tracklets == query_tracklets[i]).flatten()
            bad_index = np.argwhere(query_cams==query_cams[i]).flatten() # cam id is the same as the topk, but is not the same track
            ignore_index = np.setdiff1d(bad_index, good_index)
            q_q_sim[i, good_index] = 1.0
            q_q_sim[i, ignore_index] = q_q_sim[i, ignore_index] - 0.5

        g_g_sim = g_g_sim.data.cpu().numpy()


        q_g_sim = score_total.copy()

        for i in range(num_gallery):
            good_index = np.argwhere(gallery_tracklets==gallery_tracklets[i]).flatten()
            bad_index = np.argwhere(gallery_cams==gallery_cams[i]).flatten() # cam id is the same as the topk, but is not the same track
            ignore_index = np.setdiff1d(bad_index, good_index)
            g_g_sim[i, good_index] = 1.0
            g_g_sim[i, ignore_index] = - 1.0

        q_g_sim[q_g_sim>1.0] = 1.0
        q_g_sim[q_g_sim<-1.0] = -1.0
        g_g_sim[g_g_sim>1.0] = 1.0
        g_g_sim[g_g_sim<-1.0] = -1.0

        k1, k2, lambda_value = 80, 15,  0.6
        print('using k-reciprocal re-ranking...')
        print('k1 ', k1, ' k2 ', k2, 'lambda_value', lambda_value)
        final_dist = re_ranking(q_g_sim, q_q_sim, g_g_sim, k1, k2, lambda_value)
        score = {'score_total':final_dist}
        scipy.io.savemat('rerank_score.mat', score)
        #!!!!!!!!!!!!!!!!!!!!! after rerank sim become dist !!!!!!!!!!!!!!!!!!!
        for i in range(num_query):
            q_cam = query_cams[i]
            ignore_index = np.argwhere(gallery_cams==q_cam).flatten()
            final_dist[i,ignore_index] = final_dist[i,ignore_index] + 2.0


        final_dist = torch.FloatTensor(final_dist)
        final_dist = final_dist.cuda()
        final_dist_clone = final_dist.clone()

        _, final_rank = final_dist.topk(k=100, dim=-1, largest=False, sorted=True)
        all_index = final_rank.data.cpu().numpy()
        return all_index
