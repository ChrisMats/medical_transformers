import os
import sys
import pdb
import json
import math
import h5py
import torch
import pickle
import shutil
import random
import inspect
import warnings
import subprocess
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from pprint import pprint
from tabulate import tabulate
import matplotlib.pylab as plt
from collections import OrderedDict
from easydict import EasyDict as edict

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
def check_dir(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.makedirs(path)
        
def dir_path(path):
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return string
    else:
        raise NotAFileError(path)        
        
def get_parent_path(path):
    return os.path.abspath(os.path.join(path, os.pardir))
    
def save_json(data, fname):
    fname = os.path.abspath(fname)
    if not fname.endswith('.json'):
        fname += '.json'
    with open(fname, 'w') as wfile:  
        json.dump(data, wfile)
        
def load_json(fname):
    fname = os.path.abspath(fname)
    with open(fname, "r") as rfile:
        data = json.load(rfile)
    return data

def save_pickle(data, fname):
    fname = os.path.abspath(fname)
    if not fname.endswith('.pickle'):
        fname += '.pickle'    
    with open(fname, 'wb') as wfile:
        pickle.dump(data, wfile, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_pickle(fname):
    fname = os.path.abspath(fname)
    with open(fname, 'rb') as rfile:
        data = pickle.load(rfile)
    return data      

def get_saved_model_path(checkpoint_name):
    path = os.path.join(os.getcwd(), 'checkpoints')
    if not os.path.exists(path):
        raise IOError("Checkpoint path {} does not exist".format(path))
    else:
        return os.path.join(path, checkpoint_name) 
    
def load_params(args):
    if args.checkpoint:
        checkpoint_path = get_saved_model_path(args.checkpoint)
        checkpoint = torch.load(checkpoint_path)
        checkpoint['parameters']['training_params']['model_name'] = args.checkpoint
        return checkpoint['parameters']    
    elif args.params_path:
        return load_json(args.params_path)

    raise IOError("Please define the training paramenters")
        
def isnan(x):
    return x != x

def iszero(x):
    return x == 0    
