""" Pytorch Distributed utils
    This piece of code was heavily inspired by the equivalent of Fairseq-py
    https://github.com/pytorch/fairseq
"""


from __future__ import print_function

import datetime
import math
import pickle
import torch.distributed

from onmt.utils.logging import logger


def is_master(opt, device_id):
    return opt.gpu_ranks[device_id] == 0


def multi_init(opt, device_id):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip=opt.master_ip,
        master_port=opt.master_port)
    dist_world_size = opt.world_size
    print("in distributed.py multi_init before torch.distributed.init_process_group")
    print("dist_init_method  = ",dist_init_method)
    print("opt.gpu_ranks=",opt.gpu_ranks[device_id])
    print("device_id = ",device_id)
    print("opt.gpu_ranks[device_id]=",opt.gpu_ranks[device_id])
    torch.distributed.init_process_group(
        backend=opt.gpu_backend, init_method=dist_init_method,
        world_size=dist_world_size, rank=opt.gpu_ranks[device_id]
        ,timeout=datetime.timedelta(0,1800)
    )
    # timeout 1800 seconds

    gpu_rank = torch.distributed.get_rank()
    print("in distributed.py multi_init current gpu = ",gpu_rank)
    if not is_master(opt, device_id):
        logger.disabled = True

    return gpu_rank


def all_reduce_and_rescale_tensors(tensors, rescale_denom,
                                   buffer_size=265390320 * 20+1000):
    print("in all_reduce_and_rescale_tensors")
    logger.info("in all_reduce_and_rescale_tensors")

    """All-reduce and rescale tensors in chunks of the specified size.

    Args:
        tensors: list of Tensors to all-reduce
        rescale_denom: denominator for rescaling summed Tensors
        buffer_size: all-reduce chunk size in bytes
        buffer_size原来的值是buffer_size=10485760，后来被提高了
    """
    # buffer size in bytes, determine equiv. # of elements based on data type
    buffer_t = tensors[0].new(
        math.ceil(buffer_size / tensors[0].element_size())).zero_()
    '''
    tensor.new()制造一个跟原来的tensor有相同的datatype的tensor
    '''
    buffer = []

    def all_reduce_buffer():
        # copy tensors into buffer_t
        offset = 0
        for t in buffer:
            numel = t.numel()
            buffer_t[offset:offset+numel].copy_(t.view(-1))
            offset += numel

        # all-reduce and rescale
        torch.distributed.all_reduce(buffer_t[:offset])
        buffer_t.div_(rescale_denom)

        # copy all-reduced buffer back into tensors
        offset = 0
        for t in buffer:
            numel = t.numel()
            t.view(-1).copy_(buffer_t[offset:offset+numel])
            offset += numel

    filled = 0
    zq_total_B=0
    for t in tensors:
        sz = t.numel() * t.element_size()
        zq_total_B += sz
        '''
        element_size() 返回单个元素的字节大小。 例： >>> torch.FloatTensor().element_size() 4 >>> torch.ByteTensor().element_size() 1
        numel() calculate the number of elements in a tensor
        '''
        if sz > buffer_size:
            # tensor is bigger than buffer, all-reduce and rescale directly
            print("tensor is bigger than buffer, all-reduce and rescale directly")
            logger.info("tensor is bigger than buffer, all-reduce and rescale directly")

            print("sz= ",sz , ", buffer_size=",buffer_size)
            logger.info("sz=%d , buffer_size=%d"%(sz,buffer_size))

            print("before torch.distributed.all_reduce")
            logger.info("before torch.distributed.all_reduce")

            torch.distributed.all_reduce(t) #所以多机reduce梯度的核心还是在于torch.distributed.all_reduce()这个pytorch的封装好的函数
            print("finish torch.distributed.all_reduce")
            logger.info("finish torch.distributed.all_reduce")

            t.div_(rescale_denom)
        elif filled + sz > buffer_size:
            # buffer is full, all-reduce and replace buffer with grad
            print("buffer is full, all-reduce and replace buffer with grad")
            logger.info("buffer is full, all-reduce and replace buffer with grad")

            logger.info("sz=%d , buffer_size=%d" % (sz, buffer_size))
            all_reduce_buffer()
            buffer = [t]
            filled = sz
        else:
            # add tensor to buffer
            buffer.append(t)
            filled += sz


    '''
    即便走完上面那个for循环之后还是小于buffer_size，那么也要进行reduce操作了。
    '''
    if len(buffer) > 0:
        all_reduce_buffer()

    print("zq_total_B = ",zq_total_B)
    logger.info("zq_total_B=%d" % (zq_total_B)) #265390320,265390320,265390320,265390320


def all_gather_list(data, max_size=4096):
    """Gathers arbitrary data from all nodes into a list."""
    world_size = torch.distributed.get_world_size()
    if not hasattr(all_gather_list, '_in_buffer') or \
            max_size != all_gather_list._in_buffer.size():
        if torch.cuda.is_available(): #modified by zhengquan for cuda is not available
            all_gather_list._in_buffer = torch.cuda.ByteTensor(max_size)
            all_gather_list._out_buffers = [
                torch.cuda.ByteTensor(max_size)
                for i in range(world_size)
            ]
        else:
            all_gather_list._in_buffer = torch.ByteTensor(max_size)
            all_gather_list._out_buffers = [
                torch.ByteTensor(max_size)
                for i in range(world_size)
            ]
    in_buffer = all_gather_list._in_buffer
    out_buffers = all_gather_list._out_buffers

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError(
            'encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size < 255*256
    in_buffer[0] = enc_size // 255  # this encoding works for max_size < 65k
    in_buffer[1] = enc_size % 255
    in_buffer[2:enc_size+2] = torch.ByteTensor(list(enc))

    if torch.cuda.is_available(): #modified by zhengquan for cuda is not available
        torch.distributed.all_gather(out_buffers, in_buffer.cuda())
    else:
        torch.distributed.all_gather(out_buffers, in_buffer)

    results = []
    for i in range(world_size):
        out_buffer = out_buffers[i]
        size = (255 * out_buffer[0].item()) + out_buffer[1].item()

        bytes_list = bytes(out_buffer[2:size+2].tolist())
        result = pickle.loads(bytes_list)
        results.append(result)
    return results
