# encoding: utf-8
"""
@author:  Xuanmeng Zhang
@contact: xuanmeng@zju.edu.cn
"""

import glob
import os
import re

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset
__all__ = ["AICity21"]

@DATASET_REGISTRY.register()
class AICity21(ImageDataset):
    """AICity.

    Dataset statistics:
        - identities: .
        - images: 244867 (train) + 1103 (query) + 31238 (gallery).
    """
    dataset_dir = "aicity21_all"
    dataset_name = "aicity21"

    def __init__(self, root='datasets', **kwargs):
        self.root=root
        self.dataset_dir =  os.path.join(self.root, self.dataset_dir)

        self.train_label = os.path.join(self.dataset_dir, "all_trainval_pids.txt")
        self.query_label = os.path.join(self.dataset_dir, "query_list.txt")
        self.gallery_label = os.path.join(self.dataset_dir, "gallery_list.txt")

        self.train_dir = os.path.join(self.dataset_dir, 'image_train')
        self.query_dir = os.path.join(self.dataset_dir, 'image_query')
        self.gallery_dir =os.path.join(self.dataset_dir, 'image_test')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
            self.train_label, 
            self.query_label, 
            self.gallery_label
        ]
        self.check_before_run(required_files)

        train = self.process_train()
        query, gallery = self.process_test()

        super(AICity21, self).__init__(train, query, gallery, **kwargs)
    
    def process_train(self):
        with open(self.train_label, 'r') as f:
            train_list = [i.strip('\n') for i in f.readlines()]

        train_data = []
        for data_item in train_list:
            img_path = os.path.join(self.train_dir, data_item)
            pid = data_item.split('_')[0]
            camid = data_item.split('_')[1]
            pid = self.dataset_name + "_" + pid
            camid = self.dataset_name + "_" + camid
            train_data.append([img_path, pid, camid])

        return train_data

    def process_test(self):
        with open(self.query_label, 'r') as f:
            query_list = [i.strip('\n') for i in f.readlines()]

        with open(self.gallery_label, 'r') as f:
            gallery_list = [i.strip('\n') for i in f.readlines()]

        query_data = []
        for data_item in query_list:
            img_path = os.path.join(self.query_dir, data_item)
            pid = data_item.split('.')[0]
            camid = '0'
            query_data.append([img_path, int(pid), camid])

        gallery_data = []

        for data_item in gallery_list:
            img_path = os.path.join(self.gallery_dir, data_item)
            pid = data_item.split('.')[0]
            camid = '1'
            gallery_data.append([img_path, int(pid), camid])

        return query_data, gallery_data


@DATASET_REGISTRY.register()
class AICity21Pseudo(ImageDataset):
    dataset_dir = "aicity21_all"
    dataset_name = "aicity21pseudo"

    def __init__(self, root='datasets', **kwargs):
        self.root=root
        self.dataset_dir =  os.path.join(self.root, self.dataset_dir)

        self.train_label = os.path.join(self.dataset_dir, "all_trainval_pseudo_pids.txt")
        self.query_label = os.path.join(self.dataset_dir, "query_list.txt")
        self.gallery_label = os.path.join(self.dataset_dir, "gallery_list.txt")

        self.train_dir = os.path.join(self.dataset_dir, 'image_train_with_pseudo')
        self.query_dir = os.path.join(self.dataset_dir, 'image_query')
        self.gallery_dir =os.path.join(self.dataset_dir, 'image_test')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
            self.train_label, 
            self.query_label, 
            self.gallery_label
        ]
        self.check_before_run(required_files)

        train = self.process_train()
        query, gallery = self.process_test()

        super(AICity21Pseudo, self).__init__(train, query, gallery, **kwargs)
    
    def process_train(self):
        with open(self.train_label, 'r') as f:
            train_list = [i.strip('\n') for i in f.readlines()]

        train_data = []
        for data_item in train_list:
            img_path = os.path.join(self.train_dir, data_item)
            pid = data_item.split('_')[0]
            camid = data_item.split('_')[1]
            pid = self.dataset_name + "_" + pid
            camid = self.dataset_name + "_" + camid
            train_data.append([img_path, pid, camid])

        return train_data

    def process_test(self):
        with open(self.query_label, 'r') as f:
            query_list = [i.strip('\n') for i in f.readlines()]

        with open(self.gallery_label, 'r') as f:
            gallery_list = [i.strip('\n') for i in f.readlines()]

        query_data = []
        for data_item in query_list:
            img_path = os.path.join(self.query_dir, data_item)
            pid = data_item.split('.')[0]
            camid = '0'
            query_data.append([img_path, int(pid), camid])

        gallery_data = []

        for data_item in gallery_list:
            img_path = os.path.join(self.gallery_dir, data_item)
            pid = data_item.split('.')[0]
            camid = '1'
            gallery_data.append([img_path, int(pid), camid])

        return query_data, gallery_data

@DATASET_REGISTRY.register()
class AICity21Scene1PseudoV2(ImageDataset):
    dataset_dir = "aicity21_all"
    dataset_name = "aicity21_scene1_pseudo_v2"

    def __init__(self, root='datasets', **kwargs):
        self.root=root
        self.dataset_dir =  os.path.join(self.root, self.dataset_dir)

        self.train_label = os.path.join(self.dataset_dir, "all_trainval_scene1_pseudo_v2.txt")
        self.query_label = os.path.join(self.dataset_dir, "query_list.txt")
        self.gallery_label = os.path.join(self.dataset_dir, "gallery_list.txt")

        self.train_dir = os.path.join(self.dataset_dir, 'image_train_with_scene1_pseudo_v2')
        self.query_dir = os.path.join(self.dataset_dir, 'image_query')
        self.gallery_dir =os.path.join(self.dataset_dir, 'image_test')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
            self.train_label, 
            self.query_label, 
            self.gallery_label
        ]
        self.check_before_run(required_files)

        train = self.process_train()
        query, gallery = self.process_test()

        super(AICity21Scene1PseudoV2, self).__init__(train, query, gallery, **kwargs)
    
    def process_train(self):
        with open(self.train_label, 'r') as f:
            train_list = [i.strip('\n') for i in f.readlines()]

        train_data = []
        for data_item in train_list:
            img_path = os.path.join(self.train_dir, data_item)
            pid = data_item.split('_')[0]
            camid = data_item.split('_')[1]
            pid = self.dataset_name + "_" + pid
            camid = self.dataset_name + "_" + camid
            train_data.append([img_path, pid, camid])

        return train_data

    def process_test(self):
        with open(self.query_label, 'r') as f:
            query_list = [i.strip('\n') for i in f.readlines()]

        with open(self.gallery_label, 'r') as f:
            gallery_list = [i.strip('\n') for i in f.readlines()]

        query_data = []
        for data_item in query_list:
            img_path = os.path.join(self.query_dir, data_item)
            pid = data_item.split('.')[0]
            camid = '0'
            query_data.append([img_path, int(pid), camid])

        gallery_data = []

        for data_item in gallery_list:
            img_path = os.path.join(self.gallery_dir, data_item)
            pid = data_item.split('.')[0]
            camid = '1'
            gallery_data.append([img_path, int(pid), camid])

        return query_data, gallery_data

@DATASET_REGISTRY.register()
class AICity21Back(ImageDataset):
    dataset_dir = "part_data"
    dataset_name = "aicity21_back"

    def __init__(self, root='datasets', **kwargs):
        self.root=root
        self.dataset_dir =  os.path.join(self.root, self.dataset_dir)

        self.train_label = os.path.join(self.dataset_dir, 'train_back.txt')
        self.query_label = os.path.join(self.dataset_dir, 'query_back.txt')
        self.gallery_label = os.path.join(self.dataset_dir, 'test_back.txt')

        self.train_dir = os.path.join(self.dataset_dir, 'image_train', 'back')
        self.query_dir = os.path.join(self.dataset_dir, 'image_query', 'back')
        self.gallery_dir = os.path.join(self.dataset_dir, 'image_test', 'back')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
            self.train_label, 
            self.query_label, 
            self.gallery_label
        ]
        self.check_before_run(required_files)

        train = self.process_train()
        query, gallery = self.process_test()

        super(AICity21Back, self).__init__(train, query, gallery, **kwargs)
    
    def process_train(self):
        with open(self.train_label, 'r') as f:
            train_list = [i.strip('\n') for i in f.readlines()]

        train_data = []
        for data_item in train_list:
            img_path = os.path.join(self.train_dir, data_item)
            pid = data_item.split('_')[0]
            camid = data_item.split('_')[1]
            pid = self.dataset_name + "_" + pid
            camid = self.dataset_name + "_" + camid
            train_data.append([img_path, pid, camid])

        return train_data

    def process_test(self):
        with open(self.query_label, 'r') as f:
            query_list = [i.strip('\n') for i in f.readlines()]

        with open(self.gallery_label, 'r') as f:
            gallery_list = [i.strip('\n') for i in f.readlines()]

        query_data = []
        for data_item in query_list:
            img_path = os.path.join(self.query_dir, data_item)
            pid = data_item.split('.')[0]
            camid = '0'
            query_data.append([img_path, int(pid), camid])

        gallery_data = []

        for data_item in gallery_list:
            img_path = os.path.join(self.gallery_dir, data_item)
            pid = data_item.split('.')[0]
            camid = '1'
            gallery_data.append([img_path, int(pid), camid])

        return query_data, gallery_data

@DATASET_REGISTRY.register()
class AICity21Front(ImageDataset):
    dataset_dir = "part_data"
    dataset_name = "aicity21_front"

    def __init__(self, root='datasets', **kwargs):
        self.root=root
        self.dataset_dir =  os.path.join(self.root, self.dataset_dir)

        self.train_label = os.path.join(self.dataset_dir, 'train_front.txt')
        self.query_label = os.path.join(self.dataset_dir, 'query_front.txt')
        self.gallery_label = os.path.join(self.dataset_dir, 'test_front.txt')

        self.train_dir = os.path.join(self.dataset_dir, 'image_train', 'front')
        self.query_dir = os.path.join(self.dataset_dir, 'image_query', 'front')
        self.gallery_dir = os.path.join(self.dataset_dir, 'image_test', 'front')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
            self.train_label, 
            self.query_label, 
            self.gallery_label
        ]
        self.check_before_run(required_files)

        train = self.process_train()
        query, gallery = self.process_test()

        super(AICity21Front, self).__init__(train, query, gallery, **kwargs)
    
    def process_train(self):
        with open(self.train_label, 'r') as f:
            train_list = [i.strip('\n') for i in f.readlines()]

        train_data = []
        for data_item in train_list:
            img_path = os.path.join(self.train_dir, data_item)
            pid = data_item.split('_')[0]
            camid = data_item.split('_')[1]
            pid = self.dataset_name + "_" + pid
            camid = self.dataset_name + "_" + camid
            train_data.append([img_path, pid, camid])

        return train_data

    def process_test(self):
        with open(self.query_label, 'r') as f:
            query_list = [i.strip('\n') for i in f.readlines()]

        with open(self.gallery_label, 'r') as f:
            gallery_list = [i.strip('\n') for i in f.readlines()]

        query_data = []
        for data_item in query_list:
            img_path = os.path.join(self.query_dir, data_item)
            pid = data_item.split('.')[0]
            camid = '0'
            query_data.append([img_path, int(pid), camid])

        gallery_data = []

        for data_item in gallery_list:
            img_path = os.path.join(self.gallery_dir, data_item)
            pid = data_item.split('.')[0]
            camid = '1'
            gallery_data.append([img_path, int(pid), camid])

        return query_data, gallery_data

@DATASET_REGISTRY.register()
class AICity21LeftRight(ImageDataset):
    dataset_dir = "part_data"
    dataset_name = "aicity21_left_right"

    def __init__(self, root='datasets', **kwargs):
        self.root=root
        self.dataset_dir =  os.path.join(self.root, self.dataset_dir)

        self.train_label = os.path.join(self.dataset_dir, 'train_left_right.txt')
        self.query_label = os.path.join(self.dataset_dir, 'query_left_right.txt')
        self.gallery_label = os.path.join(self.dataset_dir, 'test_left_right.txt')

        self.train_dir = os.path.join(self.dataset_dir, 'image_train', 'left_right')
        self.query_dir = os.path.join(self.dataset_dir, 'image_query', 'left_right')
        self.gallery_dir = os.path.join(self.dataset_dir, 'image_test', 'left_right')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
            self.train_label, 
            self.query_label, 
            self.gallery_label
        ]
        self.check_before_run(required_files)

        train = self.process_train()
        query, gallery = self.process_test()

        super(AICity21LeftRight, self).__init__(train, query, gallery, **kwargs)
    
    def process_train(self):
        with open(self.train_label, 'r') as f:
            train_list = [i.strip('\n') for i in f.readlines()]

        train_data = []
        for data_item in train_list:
            img_path = os.path.join(self.train_dir, data_item)
            pid = data_item.split('_')[0]
            camid = data_item.split('_')[1]
            pid = self.dataset_name + "_" + pid
            camid = self.dataset_name + "_" + camid
            train_data.append([img_path, pid, camid])

        return train_data

    def process_test(self):
        with open(self.query_label, 'r') as f:
            query_list = [i.strip('\n') for i in f.readlines()]

        with open(self.gallery_label, 'r') as f:
            gallery_list = [i.strip('\n') for i in f.readlines()]

        query_data = []
        for data_item in query_list:
            img_path = os.path.join(self.query_dir, data_item)
            pid = data_item.split('.')[0]
            camid = '0'
            query_data.append([img_path, int(pid), camid])

        gallery_data = []

        for data_item in gallery_list:
            img_path = os.path.join(self.gallery_dir, data_item)
            pid = data_item.split('.')[0]
            camid = '1'
            gallery_data.append([img_path, int(pid), camid])

        return query_data, gallery_data

@DATASET_REGISTRY.register()
class AICity21Top(ImageDataset):
    dataset_dir = "part_data"
    dataset_name = "aicity21_top"

    def __init__(self, root='datasets', **kwargs):
        self.root=root
        self.dataset_dir =  os.path.join(self.root, self.dataset_dir)

        self.train_label = os.path.join(self.dataset_dir, 'train_top.txt')
        self.query_label = os.path.join(self.dataset_dir, 'query_top.txt')
        self.gallery_label = os.path.join(self.dataset_dir, 'test_top.txt')

        self.train_dir = os.path.join(self.dataset_dir, 'image_train', 'top')
        self.query_dir = os.path.join(self.dataset_dir, 'image_query', 'top')
        self.gallery_dir = os.path.join(self.dataset_dir, 'image_test_', 'top')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
            self.train_label, 
            self.query_label, 
            self.gallery_label
        ]
        self.check_before_run(required_files)

        train = self.process_train()
        query, gallery = self.process_test()

        super(AICity21Top, self).__init__(train, query, gallery, **kwargs)
    
    def process_train(self):
        with open(self.train_label, 'r') as f:
            train_list = [i.strip('\n') for i in f.readlines()]

        train_data = []
        for data_item in train_list:
            img_path = os.path.join(self.train_dir, data_item)
            pid = data_item.split('_')[0]
            camid = data_item.split('_')[1]
            pid = self.dataset_name + "_" + pid
            camid = self.dataset_name + "_" + camid
            train_data.append([img_path, pid, camid])

        return train_data

    def process_test(self):
        with open(self.query_label, 'r') as f:
            query_list = [i.strip('\n') for i in f.readlines()]

        with open(self.gallery_label, 'r') as f:
            gallery_list = [i.strip('\n') for i in f.readlines()]

        query_data = []
        for data_item in query_list:
            img_path = os.path.join(self.query_dir, data_item)
            pid = data_item.split('.')[0]
            camid = '0'
            query_data.append([img_path, int(pid), camid])

        gallery_data = []

        for data_item in gallery_list:
            img_path = os.path.join(self.gallery_dir, data_item)
            pid = data_item.split('.')[0]
            camid = '1'
            gallery_data.append([img_path, int(pid), camid])

        return query_data, gallery_data













