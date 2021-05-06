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

def circle_loss(feats, labels, batch_size, margin=0.5, num_instances=4, gamma=5):
    N = batch_size
    sim = cosine_dist(feats, feats)
    sim = fluid.layers.reshape(sim, (N*N,))

    labels = fluid.layers.cast(labels, dtype='float32')
    labels = fluid.layers.reshape(labels, shape=[N,1])
    expand_labels = fluid.layers.expand(x=labels, expand_times=[1, N]) # N x N
    expand_labels_t = fluid.layers.transpose(expand_labels, perm=[1,0]) 

    is_pos = fluid.layers.equal(x=expand_labels, y=expand_labels_t) 
    self_mask = fluid.layers.eye(N)
    self_mask = fluid.layers.reshape(self_mask, ( N * N ,))
    is_pos = fluid.layers.cast(is_pos, dtype='int32')
    is_pos = fluid.layers.reshape(is_pos, ( N * N ,)) # one column
    is_pos = is_pos - self_mask # filter self pos, since its cos sim is 1
    is_pos = fluid.layers.cast(is_pos, dtype='bool') ### get  0,1 label
    pos_index = fluid.layers.where(is_pos) ### example: 1,2 , 11,12
    pos_index.stop_gradient = True

    is_neg = fluid.layers.not_equal(x=expand_labels, y=expand_labels_t)
    is_neg = fluid.layers.cast(is_neg, dtype='int32')
    is_neg = fluid.layers.reshape(is_neg, (N * N,))
    is_neg = fluid.layers.cast(is_neg, dtype='bool')
    neg_index = fluid.layers.where(is_neg)
    neg_index.stop_gradient = True

    sp = fluid.layers.gather(sim, pos_index)
    sn = fluid.layers.gather(sim, neg_index)

    #alpha_p = fluid.layers.clamp(-sp + 1 + margin, min=0.001)
    #alpha_n = fluid.layers.clamp(sn + margin, min=0.001)
    alpha_p = fluid.layers.clip(-sp + 1 + margin, min=0.001, max=10000.0)
    alpha_n = fluid.layers.clip(sn + margin, min=0.001, max=10000.0)
    alpha_p.stop_gradient = True
    alpha_n.stop_gradient = True
    delta_p = 1 - margin
    delta_n = margin

    logit_p = - gamma * alpha_p * (sp - delta_p)
    logit_n = gamma * alpha_n * (sn - delta_n)
    #fluid.layers.Print(logit_p, message='logit_p', summarize=-1)
    #fluid.layers.Print(logit_n)
    loss_p = fluid.layers.logsumexp(logit_p, dim=0)
    #fluid.layers.Print(loss_p, message='loss_p', summarize=-1)
    loss_n = fluid.layers.logsumexp(logit_n, dim=0)
    loss_all = loss_p + loss_n
    #fluid.layers.Print(loss_n)
    loss = fluid.layers.softplus(loss_all)
    loss = 0.5 * loss
    return loss
