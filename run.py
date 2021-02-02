import sys
import json
import argparse
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0,'src\lib')

from time import time
from src.webscraper import * 

seed = 234
np.random.seed(seed)

parser = argparse.ArgumentParser()
"""
with open('./config/model-params.json') as f:
        p = json.loads(f.read())
        epochs = p['epochs']
        dim = p['dim']
        gcn_layer = p['gcn_layer']
        lpa_iter = p['lpa_iter']
        l2_weight = p['l2_weight']
        lpa_weight = p['lpa_weight']
        dropout = p['dropout']
        lr = p['lr']

with open('./config/data-params.json') as f:
        p = json.loads(f.read())
        dataset = p['dataset']
"""
parser.add_argument('flag',nargs = '+')
#parser.add_argument('--model', type = str, default = 'GCN-LPA', help = 'which model to use')

t = time()
args = parser.parse_args()
if 'test' in args.flag or 'all' in args.flag:
    #first download all the necessary data
    getData()
    #then run the data through the model


if 'build' in args.flag or 'train' in args.flag:
    getData()

print('time used: %d s' % (time() - t))
