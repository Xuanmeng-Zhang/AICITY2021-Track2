import paddle
import paddle.fluid as fluid
import pdb


def normalize(x, axis=-1):
    x = fluid.layers.l2_normalize(x=x, axis=axis)
    return x


def euclidean_dist(x, y, batch_size):
    m, n = batch_size, batch_size
    #m, n = x.shape[0], y.shape[0]
    #fluid.layers.Print(x)
    #xx = fluid.layers.pow(x, factor=2.0)
    xx = fluid.layers.elementwise_mul(x, x)
    xx = fluid.layers.reduce_sum(xx, dim=1, keep_dim=True)
    #fluid.layers.Print(xx)
    xx = fluid.layers.expand(x=xx, expand_times=[1, n])
    
    
    #yy = fluid.layers.pow(y, factor=2.0)
    yy = fluid.layers.elementwise_mul(y, y)
    yy = fluid.layers.reduce_sum(yy, dim=1, keep_dim=True)
    yy = fluid.layers.expand(x=yy, expand_times=[1, m])
    yy = fluid.layers.transpose(yy, perm=[1,0])
    dist = xx + yy
    mul_xy = fluid.layers.matmul(x, y, False, True)
    dist -= 2*mul_xy
    #dist = fluid.layers.clip(dist, 1e-12, 400.0)
    dist = fluid.layers.clip(dist, 1e-12, 10000.0)
    dist = fluid.layers.sqrt(dist)
    return dist


def hard_example_mining(dist_mat, labels, batch_size, num_instances):
    assert len(dist_mat.shape) == 2
    #assert dist_mat.shape[0] == dist_mat.shape[1]
    N = batch_size 
    labels = fluid.layers.cast(labels, dtype='float32')
    #fluid.layers.Print(labels, message='label', summarize=-1)
    labels = fluid.layers.reshape(labels, shape=[N,1])
    #fluid.layers.Print(labels, message='unsqueeze label', summarize=-1)
    expand_labels = fluid.layers.expand(x=labels, expand_times=[1, N]) # N x N
    expand_labels_t = fluid.layers.transpose(expand_labels, perm=[1,0]) 
    
    #fluid.layers.Print(expand_labels)
    #fluid.layers.Print(expand_labels_t)
    
    is_pos = fluid.layers.equal(x=expand_labels, y=expand_labels_t) 
    #fluid.layers.Print(is_pos, message='is_pos from equal')
    is_pos = fluid.layers.cast(is_pos, dtype='int32')
    #is_pos = fluid.layers.reshape(is_pos, ( -1, )) # one column
    is_pos = fluid.layers.reshape(is_pos, ( N * N ,)) # one column
    #fluid.layers.Print(is_pos, message='is_pos reshape')

    is_pos = fluid.layers.cast(is_pos, dtype='bool') ### get  0,1 label
    pos_index = fluid.layers.where(is_pos) ### example: 1,2 , 11,12
    #fluid.layers.Print(pos_index, message='pos_index', summarize=-1)
    pos_index.stop_gradient = True


    is_neg = fluid.layers.not_equal(x=expand_labels, y=expand_labels_t)
    is_neg = fluid.layers.cast(is_neg, dtype='int32')
    is_neg = fluid.layers.reshape(is_neg, (N * N,))
    is_neg = fluid.layers.cast(is_neg, dtype='bool')

    neg_index = fluid.layers.where(is_neg)
    #fluid.layers.Print(neg_index, message='neg_index', summarize=-1)
    neg_index.stop_gradient = True
    
    #fluid.layers.Print(dist_mat)
    dist_mat = fluid.layers.reshape(dist_mat, [N * N, 1])
    #fluid.layers.Print(dist_mat)

    ap_candi = fluid.layers.gather(dist_mat, pos_index)
    #fluid.layers.Print(ap_candi, message='gather pos dis from dist_mat')
    ap_candi = fluid.layers.reshape(ap_candi, [N, num_instances])
    dist_ap = fluid.layers.reduce_max(ap_candi, dim=1, keep_dim=True)
    
    an_candi = fluid.layers.gather(dist_mat, neg_index)
    an_candi = fluid.layers.reshape(an_candi, [N, batch_size - num_instances])
    dist_an = fluid.layers.reduce_min(an_candi, dim=1, keep_dim=True)
    return dist_ap, dist_an


def tripletLoss(global_feat, labels, batch_size, margin=0.3, num_instances=4, normalize_feature=False):
    #pdb.set_trace()
    if normalize_feature:
        global_feat = normalize(global_feat)
    dist_mat = euclidean_dist(global_feat, global_feat, batch_size)
    #fluid.layers.Print(dist_mat)
    dist_ap, dist_an = hard_example_mining(dist_mat, labels, batch_size, num_instances)
    y = fluid.layers.ones(shape=dist_an.shape, dtype='float32')
    loss = fluid.layers.margin_rank_loss(y, dist_an, dist_ap, margin=margin)
    return loss


