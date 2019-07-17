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
parser.add_argument('--cls_method', type=str, default='dbscan')

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--b_sz', type=int, default=100)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--emb_size', type=int, default=2)
parser.add_argument('--max_vali_f1', type=float, default=0)
# Hyper parameters
parser.add_argument('--alpha', type=float, default=10)
parser.add_argument('--beta', type=float, default=20)
parser.add_argument('--gamma', type=float, default=0.001)
parser.add_argument('--lr', type=float, default=0.025)
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
    args.name = f'{args.name}_{args.dataSet}_{args.cls_method}_{time.strftime("%m-%d_%H-%M")}'
    args.out_path  = args.out_dir  + '/' + args.name
    if not os.path.isdir(args.out_path): os.mkdir(args.out_path)

    logger = getLogger(args.name, args.out_path, args.config_dir)
    logger.info(f'Implementation of DeepFD, all results, embeddings and loggings will be saved in {args.out_path}/')
    Dl = DataLoader(args, logger)
    device = args.device
    features = torch.FloatTensor(getattr(Dl, Dl.ds+'_u2p').toarray()).to(device)

    deepFD = DeepFD(features, features.size(1), args.hidden_size, args.emb_size)
    deepFD.to(args.device)
    model_loss = Loss_DeepFD(features, getattr(Dl, Dl.ds+'_simi'), args.device, args.alpha, args.beta, args.gamma)
    if args.cls_method == 'mlp':
        cls_model = Classification(args.emb_size)
        cls_model.to(args.device)

    for epoch in range(args.epochs):
        logger.info(f'----------------------EPOCH {epoch}-----------------------')
        deepFD = train_model(Dl, args, logger, deepFD, model_loss, device, epoch)
        if args.cls_method == 'dbscan':
            test_dbscan(Dl, args, logger, deepFD, epoch)
        elif args.cls_method == 'mlp':
            args.max_vali_f1 = train_classification(Dl, args, logger, deepFD, cls_model, device, args.max_vali_f1, epoch)

if __name__ == '__main__':
    main()