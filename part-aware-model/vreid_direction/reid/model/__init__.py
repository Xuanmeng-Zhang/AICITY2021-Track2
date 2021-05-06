from __future__ import absolute_import

from .resnet_vd import ResNet18_vd, ResNet34_vd, ResNet50_vd, ResNet101_vd, ResNet152_vd, ResNet200_vd
from .resnext101_wsl import ResNeXt101_32x8d_wsl, ResNeXt101_32x16d_wsl
from .res2net_vd import Res2Net101_vd_26w_4s, Res2Net200_vd_26w_4s
from .resnest import ResNeSt50_fast_1s1x64d, ResNeSt50
from .hrnet import HRNet_W18_C, HRNet_W48_C, SE_HRNet_W64_C
from .efficientnet import EfficientNetB4

from .resnext_vd import ResNeXt50_vd_64x4d, ResNeXt101_vd_64x4d, ResNeXt152_vd_64x4d, ResNeXt50_vd_32x4d, ResNeXt101_vd_32x4d, ResNeXt152_vd_32x4d
from .se_resnext_vd import SE_ResNeXt50_vd_32x4d, SE_ResNeXt101_vd_32x4d, SENet154_vd






from .feature_net import reid_feature_net, reid_feature_net_sbs

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr



__factory__ = {
    
    'ResNet50_vd' : ResNet50_vd,

     'ResNet101_vd': ResNet101_vd,                           #0.8017  0.9497  6.11704 13.76222    # ssld 0.8373   0.9669  6.11704 13.76222

#     'HRNet_W18_C': HRNet_W18_C,                             # ssld 0.812 0.769   0.043   7.406   13.297
     'HRNet_W48_C': HRNet_W48_C                            # ssld 0.836 0.790   0.046   13.707  34.435
    #'SE_HRNet_W64_C': SE_HRNet_W64_C,                       # ssld 0.848 -   -   31.697  

#     'ResNeXt101_32x8d_wsl':ResNeXt101_32x8d_wsl,            # 0.8255    0.9674  18.52528    34.25319
    #'ResNeXt101_32x16d_wsl':ResNeXt101_32x16d_wsl,          # 0.8424    0.9726  25.60395    71.88384

    #'Res2Net50_vd':Res2Net50_vd_26w_4s,                     # 80.64%   95.22%  19.612
    #'Res2Net101_vd':Res2Net101_vd_26w_4s,                   #ssld 0.839  0.806   0.033   8.087   17.312
    #'Res2Net200_vd':Res2Net200_vd_26w_4s,                   #ssld 0.851  0.812   0.049   14.678  32.350

#     'ResNeXt50_vd':ResNeXt50_vd_64x4d,                      #0.8012    0.9486  13.94449    18.88759
#     'ResNeXt101_vd':ResNeXt101_vd_32x4d,                    #80.33%    95.12%  24.701
    # 'ResNeXt152_vd':ResNeXt152_vd_64x4d,                    #80.72%    95.20%  35.783
    #'SE_ResNet50_vd':SE_ResNet50_vd,                        #79.52%    94.75%  10.345
    #'SE_ResNeXt50_vd':SE_ResNeXt50_vd_32x4d,                #80.24%    94.89%  15.155
    
#     'EfficientNetB4':EfficientNetB4


}

class model_creator():
    def __init__(self, backbone_name):
        self.backbone = __factory__[backbone_name]()
        #self.feature_net = reid_feature_net()
        self.feature_net1 = reid_feature_net_sbs()
        self.feature_net2 = reid_feature_net_sbs()
        self.feature_net3 = reid_feature_net_sbs()

    def net(self,input, is_train=False, class_dim=751, num_features = 512):
        backbone_feature = self.backbone.net(input=input, class_dim=class_dim)
        print(backbone_feature.shape)
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
    
    
    def net_orientation(self,input, is_train=False, class_dim = 8, num_features = 512, finetune=False):
        backbone_feature = self.backbone.net(input=input)
        orientation_cls, reid_feature = self.feature_net1.net(input=backbone_feature, is_train=is_train, 
                                    num_features=num_features, class_dim=class_dim, finetune=finetune)
        return orientation_cls, reid_feature
    
    
    

    # def net_mt(self,input, is_train=False, cam_class = 40, color_class = 12, type_class = 11,   num_features = 512):
    #     backbone_feature = self.backbone.net(input=input)
    #     cam_cls, _ = self.feature_net.net(input=backbone_feature, is_train=is_train, 
    #                                 num_features=num_features, class_dim=cam_class)
    #     color_cls, _ = self.feature_net.net(input=backbone_feature, is_train=is_train, 
    #                                 num_features=num_features, class_dim=color_class)
    #     type_cls, _ = self.feature_net.net(input=backbone_feature, is_train=is_train, 
    #                                 num_features=num_features, class_dim=type_class)
    #     return cam_cls, color_cls, type_cls

    

    # def net_pid_color_type_feas(self,input, is_train=False, class_dim = 751, color_class = 12, type_class = 11, num_features = 512):
    #     backbone_feature = self.backbone.net(input=input)
    #     pid_cls, reid_feature = self.feature_net.net(input=backbone_feature, is_train=is_train, 
    #                                 num_features=num_features, class_dim=class_dim)
    #     color_cls, color_feature = self.feature_net.net(input=backbone_feature, is_train=is_train, 
    #                                 num_features=num_features, class_dim=color_class)
    #     type_cls, type_feature = self.feature_net.net(input=backbone_feature, is_train=is_train, 
    #                                 num_features=num_features, class_dim=type_class)
    #     return pid_cls, color_cls, type_cls, reid_feature, color_feature, type_feature

    # def net_direct_color_type(self,input, is_train=False, class_dim = 6, color_class = 12, type_class = 11, num_features = 512):
    #     backbone_feature = self.backbone.net(input=input)
    #     angle_bin1, _ = self.feature_net.net(input=backbone_feature, is_train=is_train, 
    #                                 num_features=256, class_dim=2)
    #     angle_bin2, _ = self.feature_net.net(input=backbone_feature, is_train=is_train, 
    #                                 num_features=256, class_dim=2)
    #     angle_bin3, _ = self.feature_net.net(input=backbone_feature, is_train=is_train, 
    #                                 num_features=256, class_dim=2)
    #     angle_bin4, _ = self.feature_net.net(input=backbone_feature, is_train=is_train, 
    #                                 num_features=256, class_dim=2)
    #     angle_bin5, _ = self.feature_net.net(input=backbone_feature, is_train=is_train, 
    #                                 num_features=256, class_dim=2)
    #     angle_bin6, _ = self.feature_net.net(input=backbone_feature, is_train=is_train, 
    #                                 num_features=256, class_dim=2)
        
    #     angle_cos, _ = self.feature_net.net(input=backbone_feature, is_train=is_train, 
    #                                 num_features=256, class_dim=6)
    #     angle_sin, _ = self.feature_net.net(input=backbone_feature, is_train=is_train, 
    #                                 num_features=256, class_dim=6)


    #     color_cls, _ = self.feature_net.net(input=backbone_feature, is_train=is_train, 
    #                                 num_features=256, class_dim=color_class)
    #     type_cls, _ = self.feature_net.net(input=backbone_feature, is_train=is_train, 
    #                                 num_features=256, class_dim=type_class)
    #     return angle_bin1, angle_bin2, angle_bin3, angle_bin4, angle_bin5, angle_bin6, \
    #             angle_cos, angle_sin, color_cls, type_cls

# class model_creator_att():
#     def __init__(self, backbone_name):
#         assert 'att' in backbone_name
#         self.backbone = __factory__[backbone_name]()
#         self.feature_net = reid_feature_net()

#     def conv_bn_layer(self,
#                       input,
#                       num_filters,
#                       filter_size,
#                       act=None,
#                       name=None):

#         conv = fluid.layers.conv2d(
#             input=input,
#             num_filters=num_filters,
#             filter_size=filter_size,
#             stride=1,
#             padding=(filter_size - 1) // 2,
#             groups=1,
#             act=None,
#             param_attr=ParamAttr(name=name + "_weights"),
#             bias_attr=False)
#         if name == "conv1":
#             bn_name = "bn_" + name
#         else:
#             bn_name = "bn" + name[3:]
#         conv = fluid.layers.batch_norm(input=conv,
#                                        act=act,
#                                        param_attr=ParamAttr(name=bn_name + '_scale'),
#                                        bias_attr=ParamAttr(bn_name + '_offset'),
#                                        moving_mean_name=bn_name + '_mean',
#                                        moving_variance_name=bn_name + '_variance')
#         return conv


#     def net_pid_color_type(self,input, is_train=False, class_dim = 751, color_class = 12, type_class = 11, num_features = 512):
#         backbone_feature = self.backbone.net(input=input)
#         conv_att = self.conv_bn_layer(backbone_feature, 1, 1, act='sigmoid', name='conv_att') * 5
#         conv_att = fluid.layers.expand(x=conv_att, expand_times=[1, 2048, 1, 1])
#         mul_att = fluid.layers.elementwise_mul(conv_att, backbone_feature)
#         pool_att = fluid.layers.pool2d(input=mul_att, pool_size=7, pool_type='avg', global_pooling=True, name='pool_attr_')
        
#         pid_cls, reid_feature = self.feature_net.net(input=pool_att, is_train=is_train, 
#                                     num_features=num_features, class_dim=class_dim)
#         color_cls, _ = self.feature_net.net(input=pool_att, is_train=is_train, 
#                                     num_features=num_features, class_dim=color_class)
#         type_cls, _ = self.feature_net.net(input=pool_att, is_train=is_train, 
#                                     num_features=num_features, class_dim=type_class)
#         return pid_cls, color_cls, type_cls, reid_feature
    
#     def net(self,input, is_train=False, class_dim=751, num_features = 512):
#         backbone_feature = self.backbone.net(input=input, class_dim=class_dim)
#         conv_att = self.conv_bn_layer(backbone_feature, 1, 1, act='sigmoid', name='conv_att') * 5
#         conv_att = fluid.layers.expand(x=conv_att, expand_times=[1, 2048, 1, 1])
#         mul_att = fluid.layers.elementwise_mul(conv_att, backbone_feature)
#         pool_att = fluid.layers.pool2d(input=mul_att, pool_size=7, pool_type='avg', global_pooling=True, name='pool_attr_')

#         reid_cls, reid_fea = self.feature_net.net(input=pool_att, is_train=is_train, 
#                                     num_features=num_features, class_dim=class_dim)
#         return reid_cls, reid_fea

