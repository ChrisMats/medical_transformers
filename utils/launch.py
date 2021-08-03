# adopted from detectron2 
# https://github.com/facebookresearch/detectron2

import os
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ._utils import synchronize
from .helpfuns import load_params
from .system_def import define_system_params



def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def launch(main_func, args=()):
    """
    Launch multi-gpu or distributed training.
    This function must be called on all machines involved in the training.
    It will spawn child processes (defined by ``num_gpus_per_machine`) on each machine.
    Assume everything is happening on a single node!
    """
    params, arguments = args
    
    # define world  
    define_system_params(params["system_params"])
    world_size = torch.cuda.device_count()
    params["system_params"].update({"world_size" : world_size})

    is_slurm_job = "SLURM_JOB_ID" in os.environ
    if is_slurm_job:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )

    if world_size > 1 and not is_slurm_job and not arguments.test:
        port = _find_free_port()
        dist_url = f"tcp://127.0.0.1:{port}"

        mp.spawn(
            _distributed_worker,
            nprocs=world_size,
            args=(main_func, world_size, dist_url, args),
            daemon=False,
        )
    elif world_size > 1 and is_slurm_job and not arguments.test:
        dist_url = f"tcp://{arguments.dist_url}:{arguments.port}"

        dist.init_process_group(
            backend="NCCL",
            init_method=dist_url,
            world_size=world_size,
            rank=rank
        )

        synchronize()

        torch.cuda.set_device(rank % torch.cuda.device_count())

        main_func(params, arguments)
    else:
        main_func(*args)


def _distributed_worker(rank, main_func, world_size, dist_url, args):
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    try:
        dist.init_process_group(
            backend="NCCL", init_method=dist_url, world_size=world_size, rank=rank
        )
    except Exception as e:
        raise e
    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    synchronize()

    assert world_size <= torch.cuda.device_count()
    torch.cuda.set_device(rank)

    params, _args = args
    main_func(params, _args)
