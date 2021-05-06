from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import shutil
import os
import time
import argparse
import functools
import numpy as np

import paddle.fluid as fluid

from reid.cos_anneal_learning_rate import cos_anneal_with_warmup_decay
from reid.data.reader_mt import create_readerMT
from reid.data.source.dataset import Dataset
from reid.model import model_creator
from reid.learning_rate import exponential_with_warmup_decay
from reid.loss.triplet_loss import tripletLoss
from reid.utils import load_params

from config import cfg, parse_args, print_arguments, print_arguments_dict 

def optimizer_build(cfg):
    momentum_rate = cfg.momentum
    weight_decay = cfg.weight_decay
    learning_rate = cfg.learning_rate
    lr = fluid.layers.cosine_decay(learning_rate, 1, cfg.max_iter)
    optimizer = fluid.optimizer.Momentum(
        learning_rate=lr,
        regularization=fluid.regularizer.L2Decay(cfg.weight_decay),
        momentum=cfg.momentum)
    return optimizer, lr


def build_train_program(main_prog, startup_prog, cfg):
    model = model_creator(cfg.model_arch)
    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            image = fluid.data(name='image', dtype='float32', shape=[None, 3, None, None])
            orientation = fluid.data(name='orientation', dtype='int64',   shape=[None, 1])

            data_loader = fluid.io.DataLoader.from_generator(feed_list=[image,orientation], capacity=128, use_double_buffer=True, iterable=False)

            orientation_out, reid_feature = model.net_orientation(input=image, class_dim=8, is_train=True, num_features = cfg.num_features)

            orientation_softmax_out = fluid.layers.softmax(orientation_out, use_cudnn=False)
            
            ### label smoothing
            label_one_hot = fluid.layers.one_hot(input=orientation, depth=8)
            smooth_label = fluid.layers.label_smooth(label=label_one_hot, epsilon=0.1, dtype="float32")
            orientation_cost = fluid.layers.cross_entropy(input=orientation_softmax_out, label=smooth_label, soft_label=True)
        
#             orientation_cost = fluid.layers.cross_entropy(input=orientation_softmax_out, label=orientation, ignore_index=-1)
            orientation_cost = fluid.layers.reduce_mean(orientation_cost)

            orientation_acc = fluid.layers.accuracy(input=orientation_softmax_out, label=orientation, k=1)
            
            avg_cost = orientation_cost
            build_program_out = [data_loader, avg_cost,orientation_acc]

            optimizer, learning_rate = optimizer_build(cfg)
            optimizer.minimize(avg_cost)
            build_program_out.append(learning_rate)

    return build_program_out




def main(cfg):
    #pdb.set_trace()
    ### only support drop_last=True
    count = 0
    #ReidDataset = Dataset(root = cfg.data_dir, split_id=0)
    #ReidDataset.load(num_val=10)

    ReidDataset = Dataset(root = cfg.data_dir)
    #ReidDataset.load_train_mt()
    ReidDataset.load_trainval_mt()
    #ReidDataset.load_trainval()
    reader_config = {'dataset':ReidDataset.train, 
                     #'img_dir':'./downsample_vehicle/images',
                     #'img_dir':'./dataset/aicity20_real/image_train',
                     'img_dir':'./dataset/aicity21_all/',
                     'batch_size':cfg.batch_size,
                     'num_instances':cfg.num_instances,
                     'sample_type':'Identity',
                     'shuffle':True,
                     'drop_last':True,
                     'worker_num':8,
                     #'worker_num':-2,
                     'use_process':True,
                     'bufsize':32,
                     'cfg':cfg,
                     'input_fields':['image','orientation']}

    devices_num = fluid.core.get_cuda_device_count()
    print("Found {} CUDA devices.".format(devices_num))

    new_reader, num_classes, num_batch_pids, num_iters_per_epoch = create_readerMT(reader_config, max_iter=cfg.max_iter*devices_num)
    #pdb.set_trace()


    assert cfg.batch_size % cfg.num_instances == 0

    #devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    #devices_num = len(devices.split(","))
    #print("Found {} CUDA devices.".format(devices_num))

    num_iters_per_epoch = int(num_iters_per_epoch / devices_num)
    print('per epoch contain iterations:', num_iters_per_epoch)
    max_epoch = int(cfg.max_iter / num_iters_per_epoch)


    cfg.train_class_num = num_classes
    print("num_pid: ", cfg.train_class_num)


    startup_prog = fluid.Program()
    train_prog = fluid.Program()


    train_reader, avg_cost, orientation_acc, lr_node = build_train_program(main_prog=train_prog, startup_prog=startup_prog, cfg=cfg)
    #avg_cost.persistable = True
    #pid_cost.persistable = True
    #color_cost.persistable = True
    #type_cost.persistable = True
    train_fetch_vars = [avg_cost, orientation_acc, lr_node]



    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(startup_prog)


    def save_model(exe, postfix, prog):
        model_path = os.path.join(cfg.model_save_dir, cfg.model_arch, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        else:
            os.makedirs(model_path)
        fluid.io.save_persistables(exe, model_path, main_program=prog)
    if cfg.pretrain:
        print(cfg.pretrain)
        #load_params(exe, train_prog, cfg.pretrain)
        def if_exist(var):
            if os.path.exists(os.path.join(cfg.pretrain, var.name)):
                print(var.name)
                return True
            else:
                return False
        fluid.io.load_vars(
            exe, cfg.pretrain, main_program=train_prog, predicate=if_exist)

    compile_program = fluid.compiler.CompiledProgram(train_prog).with_data_parallel(loss_name=avg_cost.name)
    if devices_num==1:
        places = fluid.cuda_places(0)
    else:
        places = fluid.cuda_places()
    train_reader.set_sample_list_generator(new_reader, places=places)
    train_reader.start()

    try:
        start_time = time.time()
        snapshot_loss = 0
        snapshot_time = 0

        for cur_iter in range(cfg.start_iter, cfg.max_iter):
            cur_peoch = int(cur_iter / num_iters_per_epoch)
            losses = exe.run(compile_program, fetch_list=[v.name for v in train_fetch_vars])
            cur_loss = np.mean(np.array(losses[0]))
            cur_acc = np.mean(np.array(losses[1]))
#             cur_color_loss = np.mean(np.array(losses[2]))
#             cur_type_loss = np.mean(np.array(losses[3]))
            cur_lr = np.mean(np.array(losses[2]))
            # cur_lr = np.array(fluid.global_scope().find_var('learning_rate').get_tensor())

            snapshot_loss += cur_loss

            cur_time = time.time() - start_time
            start_time = time.time()
            snapshot_time += cur_time
            #pdb.set_trace()


            output_str = '{}/{}epoch, {}/{}iter, lr:{:.6f}, loss:{:.4f}, Acc:{:.4f}, time:{} '.format(cur_peoch, max_epoch, cur_iter, cfg.max_iter, cur_lr, cur_loss,cur_acc, cur_time )
            print(output_str)
            #fluid.io.save_inference_model(cfg.model_save_dir+'/infer_model', infer_node.name, pred_list, exe, main_program=train_prog, model_filename='model', params_filename='params')

            if (cur_iter + 1) % cfg.snapshot_iter == 0:
                save_model(exe,"model_iter{}".format(cur_iter),train_prog)
                print("Snapshot {} saved, average loss: {}, \
                      average time: {}".format(
                    cur_iter + 1, snapshot_loss / float(cfg.snapshot_iter),
                    snapshot_time / float(cfg.snapshot_iter)))

                snapshot_loss = 0
                snapshot_time = 0

    except fluid.core.EOFException:
        train_reader.reset()

    save_model(exe, 'model_final', train_prog)
    print('Done!')


if __name__ == '__main__':
    args = parse_args()
    print_arguments_dict(args)
    main(args)
