from defaults import *
from utils.helpfuns import *
from .models import DINO
from self_supervised.BYOL.wrappers import BYOLWrapper


class DINOWrapper(BYOLWrapper):
    def __init__(self, parameters):
        super().__init__(parameters)

    def init_model(self):      
    # DDP broadcasts model states from rank 0 process to all other processes 
    # in the DDP constructor, you don’t need to worry about different DDP processes 
    # start from different model parameter initial values.
  
        # init model and wrap it with DINO
        if hasattr(transformers, self.model_params.backbone_type):
            student_params = deepcopy(self.model_params)
            teacher_params = deepcopy(self.model_params)
            student_params.transformers_params.update(drop_path_rate=0.1)
            student = Classifier(student_params)
            teacher = Classifier(teacher_params)
        else:
            student, teacher = Classifier(self.model_params), Classifier(self.model_params)
            
        if self.transfer_learning_params.use_pretrained:
            pretrained_path = self.transfer_learning_params.pretrained_path
            pretrained_model_name = self.transfer_learning_params.pretrained_model_name
            if not pretrained_path:
                pretrained_path = os.path.join(self.training_params.save_dir, "checkpoints")
            pretrained_path = os.path.join(pretrained_path, pretrained_model_name)
            load_from_pretrained(student.backbone, pretrained_path, strict=True)  
            
        teacher.load_state_dict(deepcopy(student.state_dict()))    
        momentum_iters = len(self.dataloaders.trainloader) * self.training_params.epochs
        model = DINO(student, teacher, momentum_iters)
        
        if ddp_is_on():
            model = DDP(model, device_ids=[self.device_id])        
        return model
