from defaults import *
from utils.helpfuns import *
from .models import BYOL


class BYOLWrapper(DefaultWrapper):
    def __init__(self, parameters, use_momentum=True):
        super().__init__(parameters)
        self.is_supervised = False
        self.use_momentum = use_momentum

    def init_model(self):      
    # DDP broadcasts model states from rank 0 process to all other processes 
    # in the DDP constructor, you don’t need to worry about different DDP processes 
    # start from different model parameter initial values.
  
        # init model and wrap it with BYOL
        online_encoder = Classifier(self.model_params)
        target_encoder = Classifier(self.model_params)
        if not online_encoder.pretrained:
            online_encoder.init_with_kaiming(init_type='normal')
        if self.transfer_learning_params.use_pretrained:
            pretrained_path = self.transfer_learning_params.pretrained_path
            pretrained_model_name = self.transfer_learning_params.pretrained_model_name
            if not pretrained_path:
                pretrained_path = os.path.join(self.training_params.save_dir, "checkpoints")
            pretrained_path = os.path.join(pretrained_path, pretrained_model_name)
            load_from_pretrained(online_encoder, pretrained_path, strict=True)         

        target_encoder.load_state_dict(deepcopy(online_encoder.state_dict())) 
        momentum_iters = len(self.dataloaders.trainloader) * self.training_params.epochs
        model = BYOL(online_encoder, target_encoder, momentum_iters, use_momentum=self.use_momentum)  
        if ddp_is_on():
            model = DDP(model, device_ids=[self.device_id])
        return model
    
    def init_criteria(self):          
        # define criteria
        crit = None
        return crit
