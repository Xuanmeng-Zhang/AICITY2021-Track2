from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr


import os
import pdb


def GemPooling(feat, norm=3.0):
    feat = fluid.layers.clip(feat, min=1e-6, max=1e8)
    feat = fluid.layers.pow(feat, norm)
    feat = fluid.layers.pool2d(feat, pool_type='avg', global_pooling=True)
    feat = fluid.layers.pow(feat, 1.0/norm)
    return feat


#def GemPoolingP(feat, init_value=3.0):
#    norm = fluid.layers.create_parameter([1], 
#                                  dtype='float32', 
#                                  attr=fluid.param_attr.ParamAttr(initializer=fluid.initializer.ConstantInitializer(value = init_value)))
#                                  #name='PoolNorm', 
#                                  #attr=fluid.param_attr.ParamAttr(initializer=fluid.initializer.ConstantInitializer(value = init_value)))
#    feat = fluid.layers.clip(feat, min=1e-6, max=1e8)
#    feat = fluid.layers.pow(feat, norm)
#    feat = fluid.layers.pool2d(feat, pool_type='avg', global_pooling=True)
#    feat = fluid.layers.pow(feat, 1.0/norm)
#    return feat


class feature_net_sbs_circle():
    def __init__(self, type='sbs'):
        self.type = type

    def net(self, input, is_train = False, num_features = 256, class_dim = 751, finetune=False):
        feat = GemPooling(input, 3.0)
        reid_bn1 = fluid.layers.batch_norm(input=feat,
                                  is_test = not is_train,
                                  name = 'reid_bn1',
                                  param_attr = fluid.ParamAttr(name='reid_bn1_scale'),
                                  bias_attr = fluid.ParamAttr(name='reid_bn1_offset'),
                                  moving_mean_name='reid_bn1_moving_mean',
                                  moving_variance_name='reid_bn1_moving_variance')


        stdv = 1.0 / math.sqrt(reid_bn1.shape[1]*1.0)
        reid_fc1 = fluid.layers.fc(input = reid_bn1,
                                    size = num_features,
                                    name = 'reid_fc1',
                                    param_attr=fluid.param_attr.ParamAttr(initializer=fluid.initializer.Uniform(-stdv, stdv)))
        

        # need test if this works
        #reid_fc1 = fluid.layers.leaky_relu(x=reid_fc1, alpha=0.1) 

        # batch_size, num_features
        reid_bn2 = fluid.layers.batch_norm(input=reid_fc1,
                                  is_test = not is_train,
                                  name = 'reid_bn2',
                                  param_attr = fluid.ParamAttr(name='reid_bn2_scale'),
                                  bias_attr = fluid.ParamAttr(name='reid_bn2_offset'),
                                  moving_mean_name='reid_bn2_moving_mean',
                                  moving_variance_name='reid_bn2_moving_variance',)

        reid_norm = fluid.layers.sqrt(fluid.layers.reduce_sum(fluid.layers.square(reid_bn2), dim=1))
        reid_after_norm = fluid.layers.elementwise_div(reid_bn2, reid_norm, axis=0)

        weight = fluid.layers.create_parameter(
                shape=[class_dim, num_features],
                dtype='float32',
                name='weight_norm',
                attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Xavier()))
        weight_norm = fluid.layers.sqrt(fluid.layers.reduce_sum(fluid.layers.square(weight), dim=1))
        weight = fluid.layers.elementwise_div(weight, weight_norm, axis=0)
        weight = fluid.layers.transpose(weight, perm = [1, 0])
        reid_cls = fluid.layers.matmul(reid_after_norm, weight)


        return reid_cls, reid_bn2


class feature_net_sbs():
    def __init__(self, type='sbs'):
        self.type = type

    def net(self, input, is_train = False, num_features = 256, class_dim = 751, finetune=False):
        feat = GemPooling(input, 3.0)
        reid_bn1 = fluid.layers.batch_norm(input=feat,
                                  is_test = not is_train,
                                  name = 'reid_bn1',
                                  param_attr = fluid.ParamAttr(name='reid_bn1_scale'),
                                  bias_attr = fluid.ParamAttr(name='reid_bn1_offset'),
                                  moving_mean_name='reid_bn1_moving_mean',
                                  moving_variance_name='reid_bn1_moving_variance')


        stdv = 1.0 / math.sqrt(reid_bn1.shape[1]*1.0)
        reid_fc1 = fluid.layers.fc(input = reid_bn1,
                                    size = num_features,
                                    name = 'reid_fc1',
                                    param_attr=fluid.param_attr.ParamAttr(initializer=fluid.initializer.Uniform(-stdv, stdv)))

        # need test if this works
        #reid_fc1 = fluid.layers.leaky_relu(x=reid_fc1, alpha=0.1) 

        reid_bn2 = fluid.layers.batch_norm(input=reid_fc1,
                                  is_test = not is_train,
                                  name = 'reid_bn2',
                                  param_attr = fluid.ParamAttr(name='reid_bn2_scale'),
                                  bias_attr = fluid.ParamAttr(name='reid_bn2_offset'),
                                  moving_mean_name='reid_bn2_moving_mean',
                                  moving_variance_name='reid_bn2_moving_variance',)


        stdv = 1.0 / math.sqrt(reid_bn2.shape[1]*1.0)
        if not finetune:
            reid_cls = fluid.layers.fc(input = reid_bn2,
                                    size = class_dim,
                                    name = 'reid_cls',
                                    param_attr=fluid.param_attr.ParamAttr(initializer=fluid.initializer.Uniform(-stdv, stdv)))
        else:                        
            reid_cls = fluid.layers.fc(input = reid_bn2,
                                    size = class_dim,
                                    name = 'reid_cls',
                                    param_attr=fluid.param_attr.ParamAttr(initializer=fluid.initializer.Uniform(-stdv, stdv), learning_rate=3.0))

        return reid_cls, reid_bn2



class feature_net():
    def __init__(self, type='ori'):
        self.type = type

    def net(self, input, is_train = False, num_features = 256, class_dim = 751, finetune=False):
        input = fluid.layers.pool2d(input=input, pool_type="avg", global_pooling=True)
        reid_bn1 = fluid.layers.batch_norm(input=input,
                                  is_test = not is_train,
                                  name = 'reid_bn1',
                                  param_attr = fluid.ParamAttr(name='reid_bn1_scale'),
                                  bias_attr = fluid.ParamAttr(name='reid_bn1_offset'),
                                  moving_mean_name='reid_bn1_moving_mean',
                                  moving_variance_name='reid_bn1_moving_variance')

        # need test if this works
        reid_bn1 = fluid.layers.leaky_relu(x=reid_bn1, alpha=0.1)


        stdv = 1.0 / math.sqrt(reid_bn1.shape[1]*1.0)
        reid_fc1 = fluid.layers.fc(input = reid_bn1,
                                    size = num_features,
                                    name = 'reid_fc1',
                                    param_attr=fluid.param_attr.ParamAttr(initializer=fluid.initializer.Uniform(-stdv, stdv)))

        reid_bn2 = fluid.layers.batch_norm(input=reid_fc1,
                                  is_test = not is_train,
                                  name = 'reid_bn2',
                                  param_attr = fluid.ParamAttr(name='reid_bn2_scale'),
                                  bias_attr = fluid.ParamAttr(name='reid_bn2_offset'),
                                  moving_mean_name='reid_bn2_moving_mean',
                                  moving_variance_name='reid_bn2_moving_variance',)


        stdv = 1.0 / math.sqrt(reid_bn2.shape[1]*1.0)
        if not finetune:
            reid_cls = fluid.layers.fc(input = reid_bn2,
                                    size = class_dim,
                                    name = 'reid_cls',
                                    param_attr=fluid.param_attr.ParamAttr(initializer=fluid.initializer.Uniform(-stdv, stdv)))
        else:                        
            reid_cls = fluid.layers.fc(input = reid_bn2,
                                    size = class_dim,
                                    name = 'reid_cls',
                                    param_attr=fluid.param_attr.ParamAttr(initializer=fluid.initializer.Uniform(-stdv, stdv), learning_rate=3.0))

        return reid_cls, reid_bn2

def reid_feature_net():
    return feature_net()

def reid_feature_net_sbs():
    return feature_net_sbs()
