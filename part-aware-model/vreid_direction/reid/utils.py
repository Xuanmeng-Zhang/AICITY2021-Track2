import os
import paddle
import paddle.fluid as fluid

def load_params(exe, prog, path, ignore_params=None):
    """
    Load model from the given path.
    Args:
        exe (fluid.Executor): The fluid.Executor object.
        prog (fluid.Program): load weight to which Program object.
        path (string): URL string or loca model path.
        ignore_params (list): ignore variable to load when finetuning.
            It can be specified by finetune_exclude_pretrained_params
            and the usage can refer to the document
            docs/advanced_tutorials/TRANSFER_LEARNING.md
    """
    if not (os.path.isdir(path) or os.path.exists(path + '.pdparams')):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))

    #logger.info(logger.coloring('Loading parameters from {}...'.format(path), 'HEADER'))

    ignore_set = set()
    state = fluid.io.load_program_state(path)

    # ignore the parameter which mismatch the shape
    # between the model and pretrain weight.
    all_var_shape = {}
    for block in prog.blocks:
        for param in block.all_parameters():
            all_var_shape[param.name] = param.shape
    ignore_set.update([
        name for name, shape in all_var_shape.items()
        if name in state and shape != state[name].shape
    ])

    if ignore_params:
        all_var_names = [var.name for var in prog.list_vars()]
        ignore_list = filter(
            lambda var: any([re.match(name, var) for name in ignore_params]),
            all_var_names)
        ignore_set.update(list(ignore_list))

    if len(ignore_set) > 0:
        for k in ignore_set:
            if k in state:
                #logger.warning('variable {} is already excluded automatically'.format(k))
                del state[k]
    #for k in state:
    #    print(k, state[k].shape)
    #exit()
    fluid.io.set_program_state(prog, state)


def load(exe, prog, path, ignore_params=None, resume=False):
    if resume:
        fluid.load(prog, path, exe)
    else:
        load_params(exe, prog, path)
