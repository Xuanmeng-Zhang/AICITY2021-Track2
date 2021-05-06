from __future__ import absolute_import

from .resnet_vd import ResNet18_vd, ResNet34_vd, ResNet50_vd, ResNet101_vd, ResNet152_vd, ResNet200_vd
from .res2net_vd import Res2Net101_vd_26w_4s, Res2Net200_vd_26w_4s, Res2Net50_vd_26w_4s
from .hrnet import HRNet_W18_C, HRNet_W48_C, SE_HRNet_W64_C
from .feature_net import reid_feature_net, reid_feature_net_sbs 

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr



__factory__ = {

    'ResNet101_vd': ResNet101_vd,                           #0.8017  0.9497  6.11704 13.76222    # ssld 0.8373   0.9669  6.11704 13.76222

    'HRNet_W18_C': HRNet_W18_C,                             # ssld 0.812 0.769   0.043   7.406   13.297
    'HRNet_W48_C': HRNet_W48_C,                             # ssld 0.836 0.790   0.046   13.707  34.435

    'Res2Net50_vd':Res2Net50_vd_26w_4s,                     #ssld 0.831  0.798   0.033   4.527
    'Res2Net101_vd':Res2Net101_vd_26w_4s,                   #ssld 0.839  0.806   0.033   8.087   17.312
    'Res2Net200_vd':Res2Net200_vd_26w_4s,                   #ssld 0.851  0.812   0.049   14.678  32.350

}

class model_creator():
    def __init__(self, backbone_name):
        self.backbone = __factory__[backbone_name]()
        self.feature_net1 = reid_feature_net_sbs()
        self.feature_net2 = reid_feature_net_sbs()
        self.feature_net3 = reid_feature_net_sbs()

    def net(self,input, is_train=False, class_dim=751, num_features = 512):
        backbone_feature = self.backbone.net(input=input, class_dim=class_dim)
        reid_cls, reid_fea = self.feature_net1.net(input=backbone_feature, is_train=is_train, 
                                    num_features=num_features, class_dim=class_dim)
        return reid_cls, reid_fea

    def net_pid_color_type(self,input, is_train=False, class_dim = 751, color_class = 12, type_class = 11, num_features = 512, finetune=False):
        backbone_feature = self.backbone.net(input=input)
        pid_cls, reid_feature = self.feature_net1.net(input=backbone_feature, is_train=is_train, 
                                    num_features=num_features, class_dim=class_dim, finetune=finetune)
        color_cls, _ = self.feature_net2.net(input=backbone_feature, is_train=is_train, 
                                    num_features=num_features, class_dim=color_class)
        type_cls, _ = self.feature_net3.net(input=backbone_feature, is_train=is_train, 
                                    num_features=num_features, class_dim=type_class)
        return pid_cls, color_cls, type_cls, reid_feature
