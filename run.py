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
from src.model import *
from src.FCN import *
from src.featureSpaceDay import *

seed = 234
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('flag',nargs = '+')
#parser.add_argument('--model', type = str, default = 'GCN-LPA', help = 'which model to use')

t = time()
args = parser.parse_args()
if 'test' in args.flag:
    modelRun()
if 'all' in args.flag:
    getData()
    modelRun()


    #then run the data through the model


if 'build' in args.flag:
    getData()

print('time used: %d s' % (time() - t))
