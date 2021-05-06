# encoding: utf-8
"""
@author:  Xuanmeng Zhang
@contact: xuanmeng@zju.edu.cn
"""

import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import pickle

from fastreid.evaluation import ReidEvaluator
from fastreid.evaluation.query_expansion import aqe
from fastreid.utils import comm
from fastreid.utils.compute_dist import build_dist

logger = logging.getLogger("fastreid.aicity_submission")

def loadpickle(pickle_path):
    pf = open(pickle_path, 'rb')
    data = pickle.load(pf, encoding='latin1')
    pf.close()
    return data

def save_pickle(pickle_path, data):
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)



class AICityEvaluator(ReidEvaluator):
    def process(self, inputs, outputs):
        self.pids.extend(inputs["targets"])
        self.camids.extend(inputs["camids"])
        self.features.append(outputs.cpu())

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            features = comm.gather(self.features)
            features = sum(features, [])

            pids = comm.gather(self.pids)
            pids = sum(pids, [])

            # fmt: off
            if not comm.is_main_process(): return {}
            # fmt: on
        else:
            features = self.features
            pids = self.pids

        features = torch.cat(features, dim=0)
        # query feature
        query_features = features[:self._num_query]
        # gallery features
        gallery_features = features[self._num_query:]

        query_features = F.normalize(query_features, dim=1)
        gallery_features = F.normalize(gallery_features, dim=1)

     
        save_pkl = True
        if save_pkl:
            postfix = 'res101-ibn'
            
            query_dict = {}
            gallery_dict = {}
            for i in range(query_features.shape[0]):
                query_name = str(i+1).zfill(6)+'.jpg'
                query_dict[query_name] = query_features[i].cpu().numpy()
            
            for i in range(gallery_features.shape[0]):
                gallery_name = str(i+1).zfill(6)+'.jpg'
                gallery_dict[gallery_name] = gallery_features[i].cpu().numpy()
            
            save_pickle(os.path.join(self.cfg.OUTPUT_DIR, 'real_query_fea_' + postfix + '.pkl'), query_dict)
            save_pickle(os.path.join(self.cfg.OUTPUT_DIR, 'real_gallery_fea_' + postfix + '.pkl'), gallery_dict)

        return {}
