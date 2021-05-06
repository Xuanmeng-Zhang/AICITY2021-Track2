import paddle
import paddle.fluid as fluid
import pdb


def normalize(x, axis=-1):
    x = fluid.layers.l2_normalize(x=x, axis=axis)
    return x

def cosine_dist(x, y):
    x = normalize(x, axis=1) # bs, fea
    y = normalize(y, axis=1) # bs, fea
    sim = fluid.layers.matmul(x, y, False, True)
    return sim

#def circle_softmax(reid_cls, labels, batch_size, class_num=1695, margin=0.2, num_instances=4, gamma=5):
def circle_softmax(reid_cls, labels, class_num=1695, margin=0.2, gamma=128.0):
    #alpha_p = fluid.layers.clamp( - reid_cls + 1 + margin, min=0.0)
    #alpha_n = fluid.layers.clamp(reid_cls + margin, min=0.0)
    alpha_p = fluid.layers.clip( - reid_cls + 1 + margin, min=0.0, max=10000.0)
    alpha_n = fluid.layers.clip(reid_cls + margin, min=0.0, max=10000.0)
    alpha_p.stop_gradient = True
    alpha_n.stop_gradient = True

    delta_p = 1 - margin
    delta_n = margin

    sp = gamma * alpha_p * (reid_cls - delta_p)
    sn = gamma * alpha_n * (reid_cls - delta_n)

    targets = fluid.layers.one_hot(labels, class_num)

    pred_class_logits = targets * sp + (1.0 - targets) * sn

    return pred_class_logits
