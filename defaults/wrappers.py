from .models import *
from .datasets import *
from utils._utils import *

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DS

try:
    from torchlars import LARS
except:
    install("torchlars")
    from torchlars import LARS

class DefaultWrapper:
    """Class that wraps everything.

    Model, optimizers, schedulers, and dataloaders are initialized in this class.

    Attributes:
        param_attributes:
            All the fields in the .json file are stored as attributes here.
    """
    def __init__(self, parameters: edict):
        """Inits the DefaultWrapper class.
        
        Args:
            parameters:
                Dictionary of paramaters read from a .json file.
        """
        super().__init__()
        self.is_supervised = True
        parameters = edict(deepcopy(parameters))
        parameters = self.update_augmentation_strategy(parameters)
        self.param_attributes = list(parameters.keys())
        # adding effective batch size to optimizer_params
        batch_size = parameters.dataloader_params.trainloader.batch_size
        effective_batch_size = batch_size * self.visible_world
        for key in parameters.optimization_params.keys():
            parameters.optimization_params[key]['effective_batch_size'] = effective_batch_size    
            autoscale_lr = parameters.optimization_params[key].optimizer.autoscale_lr
            if autoscale_lr:
                def_lr = parameters.optimization_params[key].optimizer.params.lr
                scaled_lr = def_lr * effective_batch_size / 256.
                parameters.optimization_params[key].optimizer.params.lr = scaled_lr            
        for key in parameters:
            setattr(self, key, parameters[key])        
        
    def instantiate(self):        
        """Initialize model, loss, metrics, dataloaders, optimizer and scheduler."""
        if self.is_rank0:
            print("Initialising Dataloaders . . .")
                    
        self.dataloaders = self.init_dataloaders()
        img_channels = self.dataloaders.trainloader.dataset.img_channels
        n_classes = self.dataloaders.trainloader.dataset.n_classes
        knn_nhood = self.dataloaders.trainloader.dataset.knn_nhood
        target_metric = self.dataloaders.trainloader.dataset.target_metric
        print_ddp(f"The default metric has been set to : \033[94m{target_metric}\033[0m")
        
        self.model_params.img_channels = img_channels
        self.model_params.knn_nhood = knn_nhood
        self.model_params.target_metric = target_metric
        
        # Checking for binary multi-label
        self.model_params.n_classes = n_classes
        is_multiclass = self.dataloaders.trainloader.dataset.is_multiclass
        if not is_multiclass and n_classes <= 2:
            self.model_params.n_classes = 1
        
        # init and get model
        print_ddp("Initialising Model . . .")        
        self.model = self.init_model()  
        
        print_ddp("Initialising Optimization methods . . ")                
        # init and get optimizer
        optimizer_defs = self.init_optimizer(self.model, self.optimization_params.default)  
        self.attr_from_dict(optimizer_defs)
        
        # init and get scheduler
        epochs = self.training_params.epochs
        scheduler_defs = self.init_scheduler(self.optimizer,
                                              self.optimization_params.default, 
                                              len(self.dataloaders.trainloader), 
                                              epochs)  
        self.schedulers = MixedLRScheduler(**scheduler_defs)
        
        # init loss functions
        self.criterion = self.init_criteria()  
        
        # init metric functions
        self.init_metrics()
        
    def init_dataloaders(self, collate_fn=None) -> edict:
        """Define dataset params and dataloaders.
        
        Args:
            collate_fn:
                Specific collate_fn for the torch.utils.data.DataLoader.
        
        Returns:
            A dict (EasyDict) with train, validation and test loaders. nonddp_trainloader is
            for the 2nd phase of SWAP training where we don't use the distributed sampler.
            
            {'trainloader': trainloader,
             'valloader': valloader,
             'testloader': testloader,
             'nonddp_trainloader':nonddp_trainloader}
        """ 
        feature_bank_set, feature_bank_Loader = None, None
        DataSet = self.dataset_mapper.get(self.dataset_params.dataset, False)
        assert DataSet, "Dataset not found - Plese select one of the following: {}".format(list(self.dataset_mapper.keys()))

        trainset = DataSet(self.dataset_params, mode='train')
        valset   = DataSet(self.dataset_params, mode='eval')
        testset  = DataSet(self.dataset_params, mode='test')

        if self.training_params.knn_eval or not self.is_supervised:
            feature_bank_set = DataSet(self.dataset_params, mode='train')
            feature_bank_set.transform = valset.transform # Use validation transform when setting up prototype vectors
            feature_bank_set.resizing = valset.resizing 
        
        if not self.is_supervised:
            trainset.num_augmentations = 2 
        
        #register task defs
        self.task = trainset.task
        self.is_multiclass = trainset.is_multiclass        
        
        
        train_sampler = None
        feature_bank_sampler = None
        train_shuffle = self.dataloader_params['trainloader']['shuffle']
        # distributed sampler 
        if ddp_is_on():        
            train_sampler = DS(trainset, num_replicas=self.visible_world, rank=self.device_id)
            if feature_bank_set is not None:
                feature_bank_sampler = DS(feature_bank_set, num_replicas=self.visible_world, shuffle=False,
                                          rank=self.device_id)
            self.dataloader_params['trainloader']['shuffle'] = False

        # define distributed samplers etc
        trainLoader = DataLoader(trainset, **self.dataloader_params['trainloader'],sampler=train_sampler)
        testLoader  = DataLoader(testset, **self.dataloader_params['testloader'])
        if len(valset) > 0 :
            valLoader   = DataLoader(valset, **self.dataloader_params['valloader'])
        else:
            valLoader = testLoader
        if feature_bank_set is not None:
            data_params_copy_feature_bank = deepcopy(self.dataloader_params['valloader'])
            data_params_copy_feature_bank['shuffle'] = False
            feature_bank_Loader = DataLoader(feature_bank_set,
                                     **data_params_copy_feature_bank ,sampler=feature_bank_sampler)
        self.dataloader_params['trainloader']['shuffle'] = train_shuffle

        if not len(valLoader):
            valLoader = testLoader            
            if self.is_rank0:
                warnings.warn("Warning... Using test set as validation set")

        return edict({'trainloader': trainLoader,
                         'valloader' : valLoader,
                         'testloader' : testLoader,
                         'fbank_loader' : feature_bank_Loader,
                         })
        

    def init_model(self) -> Classifier:
        """Initialize the model.
        
        DDP broadcasts model states from rank 0 process to all other processes 
        in the DDP constructor, you don’t need to worry about different DDP processes 
        start from different model parameter initial values.   
        """
        model =  Classifier(self.model_params)
        if self.transfer_learning_params.use_pretrained:
            pretrained_model_name = self.transfer_learning_params.pretrained_model_name
            pretrained_path = self.transfer_learning_params.pretrained_path
            if not pretrained_path:
                pretrained_path = os.path.join(self.training_params.save_dir, "checkpoints")
            pretrained_path = os.path.join(pretrained_path, pretrained_model_name)
            print_ddp("\033[1mLoading pretrained model : {}\033[0m".format(pretrained_model_name))
            load_from_pretrained(model, pretrained_path, strict=True)    
            
        model.to(self.device_id)
        if self.visible_world > 1 and torch.distributed.is_initialized():
            model = DDP(model, device_ids=[self.device_id])
        return model
    
    @staticmethod
    def init_optimizer(model, optimization_params:edict) -> edict:    
        """Initialize the optimizer.
        
        Args:
            optimization_params: EasyDict instance, read from the .json file.

        Returns:
            A dict (EasyDict) with optimizer and type keys.
            {'optimizer': optimizer (e.g. a torch.optim.Adam instance),
             'optimizer_type': optimizer_type (e.g. a string "Adam")}
        """
        optimizer_type = optimization_params.optimizer.type
        opt = optim.__dict__[optimizer_type]
        opt_params = optimization_params.optimizer.params
        optimizer = opt(DefaultWrapper.get_params_groups(model), **opt_params)

        # handling LARS
        lars_params = optimization_params.LARS_params
        effective_batch_size = optimization_params.effective_batch_size
        if lars_params.use and effective_batch_size >= lars_params.batch_act_thresh:
            print_ddp("LARS OPTIMIZER: \033[92m ACTIVE \033[0m")
            optimizer = LARS(optimizer=optimizer, eps=lars_params.eps, trust_coef=lars_params.trust_coef)
            optimizer.defaults = optimizer.optim.defaults             
        else:
            print_ddp("LARS OPTIMIZER: \033[93m INACTIVE \033[0m")
            
        return edict({"optimizer":optimizer, "optimizer_type":optimizer_type})
    
    @staticmethod
    def get_params_groups(model):
        """
        FROM: https://github.com/facebookresearch/dino/blob/main/utils.py
        It filters-out the no-grad params and it excludes weight_decay from all non-weight / non-bias tensors
        It will return 2 groups 0: regularized 1: not_regularized
        """
        regularized = []
        not_regularized = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]     
        
    @staticmethod        
    def init_scheduler(optimizer, optimization_params: edict, steps_per_epoch: int=None, epochs: int=None) -> edict:          
        """Initialize the learning rate scheduler.

        steps_per_epoch and epochs are set by the caller, they are not intended to be None.
        
        Args:
            optimization_params: EasyDict instance, read from the .json file.
        
        Returns:
            A dict (EasyDict) with scheduler and type keys.
            {'scheduler': scheduler (e.g. a torch.optim.lr_scheduler.OneCycleLR instance),
             'scheduler_type': scheduler_type (e.g. a string "OneCycleLR")}
        """
        schedulers = edict({"schedulers":[None], "scheduler_types":[None], 
                           "steps_per_epoch":steps_per_epoch})
        scheduler_types = optimization_params.scheduler.type
        accepted_types = [None, "LinearWarmup", "MultiStepLR", 
                               "ReduceLROnPlateau", "OneCycleLR", "CosineAnnealingLR"] 
        if not isinstance(scheduler_types, list):
            scheduler_types = [scheduler_types]        
        
        for scheduler_type in scheduler_types:
            if scheduler_type not in accepted_types:
                raise ValueError(f"{scheduler_type} is not a supported scheduler")
            
            if scheduler_type is None:
                continue
            elif scheduler_type not in optim.lr_scheduler.__dict__:
                if scheduler_type == 'LinearWarmup':
                    sch = LinearWarmup                 
                else:
                    raise NotImplementedError
            else:
                sch = optim.lr_scheduler.__dict__[scheduler_type]

            if sch.__name__ == 'OneCycleLR':
                max_lr = optimization_params.optimizer.params.lr
                sch_params = {"max_lr":max_lr, 
                              "steps_per_epoch":steps_per_epoch, 
                              "epochs":epochs,
                              "div_factor": max_lr/1e-8
                             }
                if "LinearWarmup" in scheduler_types:
                    sch_params["div_factor"] = 1.
                sch_params.update(optimization_params.scheduler.params.OneCycleLR)
            elif sch.__name__ == 'LinearWarmup':
                max_lr = optimization_params.optimizer.params.lr
                sch_params = optimization_params.scheduler.params[scheduler_type]
                sch_params.update({"max_lr":max_lr, "steps_per_epoch":steps_per_epoch})
            elif sch.__name__ == 'CosineAnnealingLR':
                T_max = steps_per_epoch * epochs
                sch_params = optimization_params.scheduler.params[scheduler_type]
                if "LinearWarmup" in scheduler_types:
                    T_max = T_max - warmup_iters
                sch_params.update({"T_max":T_max})
            else:
                sch_params = optimization_params.scheduler.params[scheduler_type]
            
            scheduler = sch(optimizer, **sch_params) 
            schedulers["schedulers"].append(scheduler)
            schedulers["scheduler_types"].append(scheduler_type)
            
            if scheduler_type == 'LinearWarmup':
                warmup_iters = scheduler.warmup_iters

        return schedulers
    
    def init_criteria(self):          
        """Initialize the loss criteria.  """
        if self.task == 'classification':
            if self.is_multiclass:
                crit = nn.CrossEntropyLoss() 
            else:
                crit = nn.BCEWithLogitsLoss() 
        else:
            raise NotImplementedError("Only classification tasks are implemented for now")
            
        return crit
    
    def init_metrics(self):
        if self.task == 'classification':
            if self.is_multiclass:
                self.metric = DefaultClassificationMetrics                
            else:
                self.metric = MultiLabelClassificationMetrics
        else:
            raise NotImplementedError("Only classification tasks are implemented for now")    
    
    def attr_from_dict(self, param_dict: edict):
        """Function that makes the dictionary key-values into attributes.
        
        This allows us to use the dot syntax. Check the .json file for the entries.

        Args:
            param_dict: The dict we populate the class attributes from.
        """
        self.name = self.__class__.__name__
        for key in param_dict:
            setattr(self, key, param_dict[key])   
            
    def update_augmentation_strategy(self, parameters):
        self_dir = os.path.dirname(os.path.abspath(inspect.getfile(self.__class__)))
        new_strategy_dir = os.path.join(self_dir, "augmentation_strategy.json")   
        if not os.path.isfile(new_strategy_dir):
            return parameters
        
        augmentation_strategy = edict(load_json(new_strategy_dir))
        general_args = augmentation_strategy.general_args
        repetition_strategy = augmentation_strategy.repetition_strategy
        transforms = augmentation_strategy.transforms
        to_change = list(transforms.keys())
        
        if not general_args.overwrite_defaults:
            return parameters
        params = deepcopy(parameters)
        
        for org_keys in parameters.dataset_params.keys():
            if org_keys in to_change:
                org_def = parameters.dataset_params[org_keys]
                updated_transforms = []
                for order, aug_type in enumerate(repetition_strategy.order):
                    new_trans = transforms[org_keys][aug_type]
                    n_augs = repetition_strategy.n_augmentations[order]
                    if general_args.inherit:
                        for key in general_args.inherit:
                            new_trans[key] = org_def[key]                    
                    for _ in range(n_augs):
                        updated_transforms.append(new_trans)
                params.dataset_params[org_keys] = updated_transforms                    
                
        return params            
        
    @property
    def parameters(self):
        return edict({key : getattr(self, key) 
                      for key in self.param_attributes})
    
    @property
    def dataset_mapper(self):
        return {
            "DDSM" : DDSM,
            "ISIC2019": ISIC2019,
            "APTOS2019": APTOS2019,
            }
    
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
