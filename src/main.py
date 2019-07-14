import os
import sys
import time
import torch
import random
import argparse
import numpy as np

from src.utils import *
from src.models import *
from src.dataLoader import *

parser = argparse.ArgumentParser(description='DeepFD')
parser.add_argument('--cuda', type=int, default=-1, help='Which GPU to run on (-1 for using CPU, 9 for not specifying which GPU to use.)')
parser.add_argument('--dataSet', type=str, default='weibo')
parser.add_argument('--file_paths', type=str, default='file_paths.json')
parser.add_argument('--config_dir', type=str, default='./configs')
parser.add_argument('--logs_dir', type=str, default='./logs')
parser.add_argument('--out_dir', default='./results')
parser.add_argument('--name', type=str, default='debug')

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--b_sz', type=int, default=100)
parser.add_argument('--emb_size', type=int, default=128)
parser.add_argument('--max_vali_f1', type=float, default=0)
args = parser.parse_args()
args.argv = sys.argv

if torch.cuda.is_available():
    if args.cuda == -1:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        device_id = torch.cuda.current_device()
        print('using device', device_id, torch.cuda.get_device_name(device_id))
args.device = torch.device(f"cuda:{args.cuda}" if args.cuda>=0 else "cpu")
if args.cuda == 9:
    args.device = torch.device('cuda')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def main():
    args.name = f'{args.name}_{args.dataSet}_{time.strftime("%m-%d_%H-%M")}'
    args.out_path  = args.out_dir  + '/' + args.name
    if not os.path.isdir(args.out_path): os.mkdir(args.out_path)

    logger = getLogger(args.name, args.out_path, args.config_dir)
    Dl = DataLoader(args, logger)

    deepFD = DeepFD(args)

if __name__ == '__main__':
    main()