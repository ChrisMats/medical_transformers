import wandb
from self_supervised.BYOL.trainer import *
        
class DINOTrainer(BYOLTrainer):
    def __init__(self, wraped_defs, freeze_last_for=1):
        super().__init__(wraped_defs) 
        self.freeze_last_for = freeze_last_for
        self.decay_scheduler = CosineSchedulerWithWarmup(
            base_value=self.optimizer.param_groups[0]["weight_decay"], 
            final_value=0.4, iters=len(self.trainloader)*self.epochs)  

        
    def global_step(self, **kwargs):
        self.optimizer.zero_grad()
        
        # get batch
        images, labels = kwargs['batch']  
        if len(labels) == 2 and isinstance(labels, list):
            ids    = labels[1]
            labels = labels[0]

        # go through the model
        with autocast(self.use_mixed_precision):
            loss = self.model(images, epoch = self.epoch-1) 
                
        # backprop
        if not self.use_mixed_precision:
            loss.backward() 
            if self.grad_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
            if self.epoch <= self.freeze_last_for:
                cancel_gradients(self.model, "student_encoder.fc.last_layer")                
            self.optimizer.step()  
        else:
            self.scaler.scale(loss).backward()
            if self.grad_clipping:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
            if self.epoch <= self.freeze_last_for:
                cancel_gradients(self.model, "student_encoder.fc.last_layer")                
            self.scaler.step(self.optimizer)
            self.scaler.update() 
        
        if self.grad_clipping:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
        self.optimizer.step() 

        if ddp_is_on():
            # assuming that all models are at the stame state 
            # Otherwise this is wrong and we need to synchronise the weights first!!!!
            self.model.module.ema_update(self.iters)
        else:
            self.model.ema_update(self.iters)

        # updating lr and wd
        self.scheduler.step(self.val_acc, self.val_loss)
        self.optimizer.param_groups[0]["weight_decay"] = self.decay_scheduler(self.iters)
        if self.iters % self.log_every == 0 or (self.iters == 1 and not self.is_grid_search):
            loss = dist_average_tensor(loss)
            if self.is_rank0:
                self.logging({'train_loss': loss.item(),
                             'learning_rate': self.get_lr()}) 
                        
    @property
    def feature_extractor(self):
        return DINO_to_classifier(self.model)
                
def DINO_to_classifier(net):
    if is_parallel(net):
        return net.module.teacher_encoder.backbone
    else:
        return net.teacher_encoder.backbone      
