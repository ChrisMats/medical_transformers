import os
import torch
import torch.distributed as dist

def get_device_type(verbose=False):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if verbose:
        print("Available GPUs:", torch.cuda.device_count())
        if use_cuda: 
            for i in range(torch.cuda.device_count()): 
                print("GPU {}: {}".format(i+1, torch.cuda.get_device_name(i)))
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    return device
    
def define_system_params(params):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    if params['use_GPU']:
        if not params['use_all_GPUs']:
            os.environ["CUDA_VISIBLE_DEVICES"] = params['which_GPUs']
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
  