import torch
import torch.distributed as dist

def is_rank0(device_id=None):
    if device_id is None:
        device_id = torch.cuda.current_device()
    return device_id == 0

def is_ddp(_class):
    return isinstance(_class, torch.nn.parallel.DistributedDataParallel)
    
def is_dp(_class):    
    return isinstance(_class, torch.nn.DataParallel)
    
def is_parallel(_class):
    return is_dp(_class) or is_ddp(_class)

def ddp_is_on():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    world_size = dist.get_world_size()
    if world_size < 2:
        return False
    return True

def print_ddp(text):
    if ddp_is_on():
        if is_rank0(torch.cuda.current_device()):
            print(text)
    else:
        print(text)

def synchronize():
    if not ddp_is_on():
        return
    dist.barrier() 

def dist_average_tensor(tensor, mode='all', dst_rank=0):
    if not ddp_is_on():
        return tensor
    world_size = float(dist.get_world_size())    
    rt = tensor.clone()
    if mode == 'all':
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    else:
        dist.reduce(rt, dst=dst_rank, op=dist.ReduceOp.SUM)  
    rt /= world_size
    return rt

def dist_gather_tensor(tensor, mode='all', dst_rank=0, concatenate=True, cat_dim=0, group=None):
    if not ddp_is_on():
        if not concatenate:
            tensor = [tensor]
        return tensor
    world_size = dist.get_world_size()  
    rt = tensor.clone()
    tensor_list = [torch.zeros_like(rt) for _ in range(world_size)]
    if mode == 'all':
        dist.all_gather(tensor_list, rt, group=group)
    else:
        if dist.get_backend() == 'nccl':
            group = dist.new_group(backend="gloo")  
        else:
            group = dist.group.WORLD            
        dist.gather(rt, 
                    gather_list=tensor_list if dist.get_rank() == dst_rank else None, 
                    dst=dst_rank, group=group)  
        if dist.get_rank() != dst_rank:
            tensor_list = [tensor]            
    if concatenate:
        tensor_list = torch.cat(tensor_list, dim=cat_dim)
    
    return tensor_list

def dist_gather_object(tensor, mode='all', dst_rank=0, concatenate=True, cat_dim=0, group=None):
    if not ddp_is_on():
        return [tensor]
    world_size = dist.get_world_size()  

    object_list = [None] * world_size
    if mode == 'all':
        dist.all_gather_object(object_list, tensor, group=group)
    else:
        object_list = [None] * world_size
        if dist.get_backend() == 'nccl':
            group = dist.new_group(backend="gloo")  
        else:
            group = dist.group.WORLD         
        dist.gather_object(tensor, 
                           object_gather_list=object_list if dist.get_rank() == dst_rank else None, 
                           dst=dst_rank, group=group)
        if dist.get_rank() != dst_rank:
            object_list = [tensor]
    
    return object_list

def dist_average_model_weights(model, mode='all'):
    if not ddp_is_on():
        return tensor
    world_size = float(dist.get_world_size())
    for param in model.parameters():
        if mode == 'all':
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        else:
            dist.reduce(param.data, dst=dst_rank, op=dist.ReduceOp.SUM)
        param.data /= world_size

def dist_gather(tensor, mode='all', dst_rank=0, concatenate=True, cat_dim=0, group=None):
    if isinstance(tensor, torch.Tensor):
        return dist_gather_tensor(tensor, mode, dst_rank, concatenate, cat_dim, group)
    else:
        return dist_gather_object(tensor, mode, dst_rank, concatenate, cat_dim, group)
    