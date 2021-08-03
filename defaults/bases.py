from utils import *

from torchvision.transforms import *
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader

from torch import nn
from torchvision import models as cnn_models
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

from torch.utils.tensorboard import SummaryWriter
import wandb
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, OneCycleLR
    
class BaseSet(Dataset):
    """Base dataset class that actual datasets, e.g. Cifar10, subclasses.

    This class only has torchvision.transforms for augmentation.
    Not intended to be used directly.
    """
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        
        img_path = self.data[idx]['img_path']
        label = torch.as_tensor(self.data[idx]['label'])
        
        png_path = '.'.join(img_path.split('.')[:-1]) + '.png'
        if os.path.exists(png_path):
            img = self.get_x(png_path)
            img_path = png_path
        else:
            img = self.get_x(img_path)

        if self.resizing is not None:
            img = self.resizing(img)

        if self.transform is not None:
            if isinstance(self.transform, list):
                img = [tr(img) for tr in self.transform]
            else:
                if self.is_multi_crop:
                    img = self.multi_crop_aug(img, self.transform)
                else:
                    img = [self.transform(img) for _ in range(self.num_augmentations)]
            img = img[0] if len(img) == 1 and isinstance(img, list) else img      
        
        return img, label
            
    def get_x(self, img_path):
        return pil_loader(img_path, self.img_channels)    
    
    def attr_from_dict(self, param_dict):
        self.is_multi_crop = False
        self.num_augmentations = 1        
        self.name = self.__class__.__name__
        for key in param_dict:
            setattr(self, key, param_dict[key])

    def get_trans_list(self, transform_dict):
        transform_list = []   
        
        if "Resize" in transform_dict:
            if transform_dict['Resize']['apply']:
                transform_list.append(Resize((transform_dict['Resize']['height'],
                                             transform_dict['Resize']['width'])))

        if "CenterCrop" in transform_dict:                
            if transform_dict['CenterCrop']['apply']:
                transform_list.append(CenterCrop((transform_dict['CenterCrop']['height'],
                                                 transform_dict['CenterCrop']['width'])))   

        if "RandomCrop" in transform_dict:                
            if transform_dict['RandomCrop']['apply']:
                padding = transform_dict['RandomCrop']['padding']
                transform_list.append(RandomCrop((transform_dict['RandomCrop']['height'],
                                             transform_dict['RandomCrop']['width']),
                                            padding=padding if padding > 0 else None))            

        if "RandomResizedCrop" in transform_dict:                
            if transform_dict['RandomResizedCrop']['apply']:
                transform_list.append(RandomResizedCrop(size=transform_dict['RandomResizedCrop']['size'],
                                                       scale=transform_dict['RandomResizedCrop']['scale'],
                                                       interpolation=InterpolationMode.BILINEAR))        

        if "VerticalFlip" in transform_dict:                
            if transform_dict['VerticalFlip']['apply']:
                transform_list.append(RandomVerticalFlip(p=transform_dict['VerticalFlip']['p']))

        if "HorizontalFlip" in transform_dict:                
            if transform_dict['HorizontalFlip']['apply']:
                transform_list.append(RandomHorizontalFlip(p=transform_dict['HorizontalFlip']['p']))

        if "RandomRotation" in transform_dict:                
            if transform_dict['RandomRotation']['apply']:
                transform_list.append(
                    rand_apply(RandomRotation(degrees=transform_dict['RandomRotation']['angle']),
                                                      p=transform_dict['RandomRotation']['p'])) 

        if "ColorJitter" in transform_dict:                
            if transform_dict['ColorJitter']['apply']:
                temp_d = transform_dict['ColorJitter']
                transform_list.append(
                    rand_apply(ColorJitter(brightness=temp_d['brightness'],
                                                  contrast=temp_d['contrast'], 
                                                  saturation=temp_d['saturation'], 
                                                  hue=temp_d['hue']),
                                                  p=temp_d['p'])) 

        if "RandomGrayscale" in transform_dict:                
            if transform_dict['RandomGrayscale']['apply']:
                transform_list.append(RandomGrayscale(p=transform_dict['RandomGrayscale']['p']))             

        if "RandomGaussianBlur" in transform_dict:                
            if transform_dict['RandomGaussianBlur']['apply']:
                transform_list.append(
                    RandomGaussianBlur(p=transform_dict['RandomGaussianBlur']['p'],
                                       radius_min=transform_dict['RandomGaussianBlur']['radius_min'],
                                       radius_max=transform_dict['RandomGaussianBlur']['radius_max'])) 

        if "RandomAffine" in transform_dict:                
            if transform_dict['RandomAffine']['apply']:
                temp_d = transform_dict['RandomAffine']
                transform_list.append(
                    rand_apply(RandomAffine(degrees=temp_d['degrees'],
                                                  translate=temp_d['translate'], 
                                                  scale=temp_d['scale'], 
                                                  shear=temp_d['shear']),
                                                  p=temp_d['p']))                    

        if "RandomPerspective" in transform_dict:                
            if transform_dict['RandomPerspective']['apply']:
                transform_list.append(RandomPerspective(transform_dict['RandomPerspective']['distortion_scale'],
                                  p=transform_dict['RandomPerspective']['p']))  

        if "RandomSolarize" in transform_dict:                
            if transform_dict['RandomSolarize']['apply']:
                transform_list.append(RandomSolarize(threshold=transform_dict['RandomSolarize']['threshold'],
                                                     p=transform_dict['RandomSolarize']['p']))              

        transform_list.append(ToTensor())
        if "Normalize" in transform_dict:            
            if transform_dict['Normalize']:
                transform_list.append(Normalize(mean=self.mean, 
                                                std=self.std)) 
                
        if "RandomErasing" in transform_dict:                    
            if transform_dict['RandomErasing']['apply']:
                temp_d = transform_dict['RandomErasing']
                transform_list.append(RandomErasing(scale=temp_d['scale'],
                                                  ratio=temp_d['ratio'], 
                                                  value=temp_d['value'],
                                                  p=temp_d['p']))             
        
        return transform_list
    
    def get_transform_defs(self):
        if self.mode == 'train':
            aplied_transforms = self.train_transforms
        if self.mode in ['val', 'eval']:
            aplied_transforms = self.val_transforms
        if self.mode == 'test':
            aplied_transforms = self.test_transforms
        return aplied_transforms
    
    def has_multi_crop(self):        
        aplied_transforms = self.get_transform_defs()
        
        if "MultiCrop" in aplied_transforms:
            self.is_multi_crop = aplied_transforms["MultiCrop"]["apply"] 
            self.multi_crop_aug = MultiCrop(n_crops=aplied_transforms["MultiCrop"]["n_crops"],
                                                    sizes=aplied_transforms["MultiCrop"]["sizes"],
                                                    scales=aplied_transforms["MultiCrop"]["scales"])
        else:
            self.is_multi_crop = False

    def get_transforms(self):   

        self.has_multi_crop() # Checking and init MultiCrop strategy
        aplied_transforms = self.get_transform_defs()
            
        if self.is_multi_crop and "RandomResizedCrop" in aplied_transforms:
            aplied_transforms["RandomResizedCrop"]["apply"] = False # turning off RandomResize in augs
        
        if isinstance(aplied_transforms, list):
            transforms = [Compose(self.get_trans_list(tr)) for tr in aplied_transforms]
        elif isinstance(aplied_transforms, dict):
            transforms = Compose(self.get_trans_list(aplied_transforms))
        else:
            raise TypeError("Transform data structure not understood")

        return self.__class__.disentangle_resizes_from_transforms(transforms)
    
    def remove_norm_transform(self):
        no_norm_transforms = deepcopy(self.transform.transforms)
        no_norm_transforms = [trans for trans in no_norm_transforms 
                              if not isinstance(trans, Normalize)]
        self.transform = Compose(no_norm_transforms)
    
    def Unormalize_image(self, image):
        norm = [trans for trans in self.transform.transforms 
                if isinstance(trans, Normalize)][0]
        unorm_mean = tuple(- np.array(norm.mean) / np.array(norm.std))
        unorm_std = tuple( 1.0 / np.array(norm.std))
        return Normalize(unorm_mean, unorm_std)(image)
    
    @staticmethod
    def remove_transform(old_transforms, transform_to_remove_type):            
        new_transforms = deepcopy(old_transforms)
        if isinstance(new_transforms, Compose):
            new_transforms = new_transforms.transforms
        new_transforms = [trans for trans in new_transforms 
                              if not isinstance(trans, transform_to_remove_type)]
        return Compose(new_transforms)   
    
    @staticmethod
    def disentangle_resizes_from_transforms(transforms):
        resizes = []
        resizing = None
        resize_free_trans = deepcopy(transforms)    
        if isinstance(transforms, Compose):
            # if it is a standard Comose of transforms
            resizing = [ tr for tr in transforms.transforms if isinstance(tr, Resize)]
            resize_free_trans = BaseSet.remove_transform(resize_free_trans, Resize)
            resizing = None if not resizing else resizing[0]
            return resize_free_trans, resizing
        elif isinstance(transforms, list):
            # if it is a list of transforms
            for ltr in transforms:
                resizes.append([ tr for tr in ltr.transforms if isinstance(tr, Resize)][0])
            sizes = [tr.size for tr in resizes]
            if len(set(sizes)) == 1 and len(sizes) > 1:
                # if all resizes are the same
                resizing = deepcopy(resizes[0])
                resize_free_trans = [BaseSet.remove_transform(tr, Resize) for tr in resize_free_trans]
                return resize_free_trans, resizing
            else:
                # if we have different resizes return the original
                return transforms, resizing
        else:
            raise TypeError(f"Resize disentaglement does not support type {type(transforms)}")    
    
    @staticmethod    
    def get_validation_ids(total_size, val_size, json_path, dataset_name, seed_n=42, overwrite=False):
        """ Gets the total size of the dataset, and the validation size (as int or float [0,1] 
        as well as a json path to save the validation ids and it 
        returns: the train / validation split)"""
        idxs = list(range(total_size))   
        if val_size < 1:
            val_size = int(total_size * val_size)  
        train_size = total_size - val_size

        if not os.path.isfile(json_path) or overwrite:
            print("Creating a new train/val split for \"{}\" !".format(dataset_name))
            random.Random(seed_n).shuffle(idxs)
            train_split = idxs[val_size:]
            val_split = idxs[:val_size]
            s_dict = {"train_split":train_split, "val_split":val_split}
            save_json(s_dict, json_path)    
        else:
            s_dict = load_json(json_path)
            if isinstance(s_dict, dict):
                val_split = s_dict["val_split"]
                train_split = s_dict["train_split"]
            elif isinstance(s_dict, list):
                val_split = s_dict
                train_split = list(set(range(total_size)) - set(val_split))
                assert len(train_split) + len(val_split) == total_size
            if val_size != len(val_split) or train_size != len(train_split):
                print("Found updated train/validation size for \"{}\" !".format(dataset_name))
                train_split, val_split = BaseSet.get_validation_ids(total_size, val_size, json_path, 
                                                          dataset_name, seed_n=42, overwrite=True)
        return train_split, val_split         
    
    
class BaseModel(nn.Module):
    """Base model that Classifier subclasses.
    
    This class only has utility functions like freeze/unfreeze and init_weights.
    Not intended to be used directly.
    """
    def __init__(self):
        super().__init__()  
        super().__init__() 
        self.use_mixed_precision = False        
        self.base_id = torch.cuda.current_device() if self.visible_world else "cpu"
    
    def attr_from_dict(self, param_dict):
        for key in param_dict:
            setattr(self, key, param_dict[key])
            
    def get_out_channels(self, m):
        def children(m): return m if isinstance(m, (list, tuple)) else list(m.children())
        c=children(m)
        if len(c)==0: return None
        for l in reversed(c):
            if hasattr(l, 'num_features'): return l.num_features
            res = self.get_out_channels(l)
            if res is not None: return res
            
    def get_submodel(self, m, min_layer=None, max_layer=None):
        return list(m.children())[min_layer:max_layer]
    
    def freeze_bn(self, submodel=None):
        submodel = self if submodel is None else submodel
        for layer in submodel.modules():
            if isinstance(layer,  nn.BatchNorm2d) or isinstance(layer,  nn.BatchNorm1d):
                layer.eval()
                
    def unfreeze_bn(self, submodel=None):
        submodel = self if submodel is None else submodel
        for layer in submodel.modules():
            if isinstance(layer,  nn.BatchNorm2d) or isinstance(layer,  nn.BatchNorm1d):
                layer.train()
                
    def freeze_submodel(self, submodel=None):
        submodel = self if submodel is None else submodel
        for param in submodel.parameters():
            param.requires_grad = False
            
    def unfreeze_submodel(self, submodel=None):
        submodel = self if submodel is None else submodel
        for param in submodel.parameters():
            param.requires_grad = True

    def initialize_norm_layers(self, submodel=None):
        submodel = self if submodel is None else submodel
        for layer in submodel.modules():
            if isinstance(layer,  nn.BatchNorm2d) or isinstance(layer,  nn.GroupNorm):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()  

    def freeze_norm_layers(self, submodel=None):
        submodel = self if submodel is None else submodel
        for layer in submodel.modules():
            if isinstance(layer,  nn.BatchNorm2d) or isinstance(layer,  nn.GroupNorm):
                layer.eval()  
                
    def unfreeze_norm_layers(self, submodel=None):
        submodel = self if submodel is None else submodel
        for layer in submodel.modules():
            if isinstance(layer,  nn.BatchNorm2d) or isinstance(layer,  nn.GroupNorm):
                layer.train()                  

    def print_trainable_params(self, submodel=None):
        submodel = self if submodel is None else submodel
        for name, param in submodel.named_parameters():
            if param.requires_grad:
                print(name)   
                
    def init_with_kaiming(self, submodel=None, init_type='normal'):
        submodel = self if submodel is None else submodel
        if init_type == 'uniform':
            weights_init = conv2d_kaiming_uniform_init
        elif init_type == 'normal':            
            weights_init = conv2d_kaiming_normal_init
        else:
            raise NotImplementedError
        submodel.apply(weights_init)             
                
    @property
    def visible_world(self):
        return torch.cuda.device_count()   
   
    @property
    def visible_ids(sefl):
        return list(range(torch.cuda.device_count()))
    
    @property
    def device_id(self):    
        did = torch.cuda.current_device() if self.visible_world else "cpu"
        assert self.base_id == did
        return did              
    
    @property
    def is_rank0(self):
        return is_rank0(self.device_id)
   
                
class BaseTrainer:
    """Base trainer class that Trainer subclasses.

    This class only has utility functions like save/load model.
    Not intended to be used directly.
    """
    def __init__(self):
        self.scaler = None
        self.use_mixed_precision = False        
        self.is_supervised = True
        self.val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.val_acc = 0.
        self.best_val_acc = 0.
        self.iters = 0
        self.epoch0 = 0
        self.epoch = 0
        self.base_id = torch.cuda.current_device() if self.visible_world else "cpu"
        self.is_grid_search = False
        self.report_intermediate_steps = True  
    
    def attr_from_dict(self, param_dict):
        for key in param_dict:
            setattr(self, key, param_dict[key])
            
    def reset(self):
        if is_parallel(self.model):
            self.model.module.load_state_dict(self.org_model_state)
            self.model.module.to(self.model.module.device_id)            
        else:
            self.model.load_state_dict(self.org_model_state)
            self.model.to(self.model.device_id)
        self.optimizer.load_state_dict(self.org_optimizer_state)
        print(" Model and optimizer are restored to their initial states ")
        
    def load_session(self, restore_only_model=False, model_path=None):
        self.get_saved_model_path(model_path=model_path)
        if os.path.isfile(self.model_path) and self.restore_session:        
            print("Loading model from {}".format(self.model_path))
            checkpoint = torch.load(self.model_path)
            if is_parallel(self.model):
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            self.model.to(self.device_id)
            self.org_model_state = model_to_CPU_state(self.model)
            self.best_model = deepcopy(self.org_model_state)
            if self.scaler is not None and "scaler" in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler'])            
            if restore_only_model:
                return
            
            self.iters = checkpoint['iters']
            self.epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device_id)
            self.org_optimizer_state = opimizer_to_CPU_state(self.optimizer)
            print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.model_path, checkpoint['epoch']))

        elif not os.path.isfile(self.model_path) and self.restore_session:
            print("=> no checkpoint found at '{}'".format(self.model_path))
    
    def get_saved_model_path(self, model_path=None):
        if model_path is None:
            if not hasattr(self, "save_dir"):
                raise AttributeError("save_dir not found. Please specify the saving directory")
            model_saver_dir = os.path.join(self.save_dir, 'checkpoints')
            check_dir(model_saver_dir)
            self.model_path = os.path.join(model_saver_dir, self.model_name)            
        else:
            self.model_path = os.path.abspath(model_path)
        
    def save_session(self, model_path=None, verbose=False):
        if self.is_rank0:
            self.get_saved_model_path(model_path=model_path)
            if verbose:
                print("Saving model as {}".format(os.path.basename(self.model_path)) )
            state = {'iters': self.iters, 'state_dict': self.best_model,
                     'optimizer': opimizer_to_CPU_state(self.optimizer), 'epoch': self.epoch,
                    'parameters' : self.parameters}
            if self.scaler is not None:
                state['scaler'] = self.scaler.state_dict()            
            torch.save(state, self.model_path)
        synchronize()
        
    def get_embedding_path(self, mode="umap_emb", iters=-1):
        self.get_saved_model_path()
        base_path, model_name = self.model_path.split("checkpoints/")
        emb_path = model_name + "-{}".format(mode)
        if iters >= 0 :
            emb_path += "_iter{}".format(iters)
        emb_path += ".png"
        emb_dir = os.path.join(base_path, "embeddings", model_name)
        check_dir(emb_dir)
        return os.path.join(emb_dir, emb_path)          
        
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
        
    def print_train_init(self):
        if self.is_rank0: 
            print("Start training with learning rate: {}".format(self.get_lr()))    
            
    def logging(self, logging_dict):
        if not self.is_rank0: return
        if self.use_tensorboard:
            for key, val in logging_dict.items():
                if isinstance(val,list):
                    for content in val:
                        self.summary_writer.add_image(key, content, self.iters,dataformats="HWC")
                else:
                    self.summary_writer.add_scalar(key, val, self.iters)
        else:
            wandb.log(logging_dict, step=self.iters)    
            
    def set_models_precision(self, use=False):
        if is_parallel(self.model):
            self.model.module.use_mixed_precision = use
        else:
            self.model.use_mixed_precision = use            
             
    @property
    def visible_world(self):
        return torch.cuda.device_count()   
   
    @property
    def visible_ids(sefl):
        return list(range(torch.cuda.device_count()))
    
    @property
    def device_id(self):    
        return torch.cuda.current_device() if self.visible_world else "cpu"
    
    @property
    def is_rank0(self):
        return is_rank0(self.device_id)       