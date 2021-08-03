#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import wandb
import argparse

import torch
from defaults import *
from utils.system_def import *
from utils.launch import dist, launch, synchronize

from self_supervised import *

global debug


def parse_arguments():
    parser = argparse.ArgumentParser(description='The main takes as \
                             argument the parameters dictionary from a json file')
    parser.add_argument('--params_path', type=str, required=False, 
                        default="./params.json",
                        help='Give the path of the json file which contains the training parameters')
    parser.add_argument('--checkpoint', type=str, required=False, help='Give a valid checkpoint name')
    parser.add_argument('--test', action='store_true', default=False, help='Flag for testing')
    parser.add_argument('--find_lr', action='store_true', default=False, help='Flag for lr finder')
    parser.add_argument('--debug', action='store_true', default=False, help='Flag for turning on the debug_mode')
    parser.add_argument('--data_location', type=str, required=False, help='Update the datapath')
    parser.add_argument('--dist_url', type=str, default='', required=False,
                        help='URL of master node, for use with SLURM')
    parser.add_argument('--port', type=int, required=False, default=45124, 
                        help='Explicit port selection, for use with SLURM')
    parser.add_argument('--gpu', type=str, required=False, help='The GPU to be used for this run')
    
    # semi/self args
    parser.add_argument('--byol', action='store_true', default=False, help='Flag for training with BYOL')
    parser.add_argument('--simsiam', action='store_true', default=False, help='Flag for training with SimSiam')
    parser.add_argument('--dino', action='store_true', default=False, help='Flag for training with DINO')
    
    return parser.parse_args()


def update_params_from_args(params, args):
    if args.gpu:
        prev_gpu = params.system_params.which_GPUs
        params.system_params.which_GPUs = args.gpu  # change the value in-place
        print('Changed GPU for this run from {} to {}'.format(prev_gpu, args.gpu))


def main(parameters, args):
    if args.data_location:
        parameters['dataset_params']['data_location'] = args.data_location
    
    # check self-supervised method
    assert not args.byol * args.simsiam, "BYOL or SimSiam can be on but not both"
    use_momentum = True if args.byol else False 
    
    # define system
    define_system_params(parameters.system_params)
    
    # Instantiate wrapper with all its definitions   
    if args.byol or args.simsiam or args.dino:
        if args.dino:
            wrapper = DINOWrapper(parameters)
        else:
            wrapper = BYOLWrapper(parameters, use_momentum=use_momentum)
    else:
        wrapper = DefaultWrapper(parameters)
    wrapper.instantiate()
    
    # initialize logger
    if wrapper.is_rank0:
        log_params = wrapper.parameters.log_params    
        training_params = wrapper.parameters.training_params
        if wrapper.log_params['run_name'] == "DEFINED_BY_MODEL_NAME":
            log_params['run_name'] = training_params.model_name  
        if args.debug:
            os.environ['WANDB_MODE'] = 'dryrun'
        if not (args.test or args.find_lr):
            if parameters.training_params.use_tensorboard:
                print("Using TensorBoard logging")
                summary_writer = SummaryWriter()
            else:
                print("Using WANDB logging")
                wandb.init(project=log_params.project_name, 
                           name=log_params.run_name, 
                           config=wrapper.parameters,
                          resume=True if training_params.restore_session else False)
    
    # define trainer 
    if args.byol or args.simsiam or args.dino:
        if args.dino:
            trainer = DINOTrainer(wrapper)
        else:
            trainer = BYOLTrainer(wrapper, use_momentum)
    else:
        trainer = Trainer(wrapper)
        
    if parameters.training_params.use_tensorboard:
        trainer.summary_writer = summary_writer
        
    if args.test:
        trainer.test()
    elif args.find_lr:
        trainer.lr_grid_search(**wrapper.parameters.lr_finder.grid_search_params)        
    else:
        trainer.train()
        if wrapper.is_supervised:
            trainer.test()
        
    
if __name__ == '__main__':
    args = parse_arguments()
    parameters = edict(load_params(args))
    update_params_from_args(parameters, args)

    try:
        launch(main, (parameters, args))
    except Exception as e:       
        if dist.is_initialized():
            dist.destroy_process_group()            
        raise e
    finally:
        if dist.is_initialized():
            synchronize()         
            dist.destroy_process_group()            
