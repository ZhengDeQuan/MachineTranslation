import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dist-rank', default=0, type=int,
                        help='rank of distributed processes')
    args = parser.parse_args()
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip="11.3.131.52",
        master_port="10111"
    )
    print("dist_init_method  = ",dist_init_method)
    dist_world_size=2
    dist.init_process_group(backend='gloo',
                            init_method=dist_init_method
                            ,world_size=dist_world_size,
                            rank=args.dist_rank)
    print("finished")

if __name__ == "__main__":
    main()