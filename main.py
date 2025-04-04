import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pipeline.model import Exp_Model
import argparse
import torch
import numpy as np
import random

fix_seed = 100
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='generating')

parser.add_argument("--train",action="store_true",help="whether to train the model")
parser.add_argument("--test",action="store_true",help="whether to test the model")
parser.add_argument("--feature_dir", type=str, default="CRT_data/texts/")
parser.add_argument("--label_dir", type=str, default="CRT_data/labels/")
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--datasets_dir", type=str, default="./CRT_data/")
parser.add_argument('--break_down',type=bool,default=False,help="whether last time training breaks down")
parser.add_argument("--save_dir", type=str, default="results/")

args = parser.parse_args()
print('Args in experiment:')
print(args)

exp_model = Exp_Model(args)
# exp_model.load_data()
if args.train:
    exp_model.train()
if args.test:
    exp_model.test()
