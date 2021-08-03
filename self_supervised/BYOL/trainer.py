import wandb
from defaults.trainer import *
        
class BYOLTrainer(Trainer):
    def __init__(self, wraped_defs, use_momentum=True):
        super().__init__(wraped_defs)
        self.use_momentum = use_momentum
        self.best_model = model_to_CPU_state(self.feature_extractor)        

    def train(self):
        self.test_mode = False
        self.print_train_init()
        
        epoch_bar = range(self.epoch0 + 1, self.epoch0 + self.epochs + 1)
        if self.is_rank0:
            epoch_bar = tqdm(epoch_bar, desc='Epoch', leave=False)
        
        # Looping thorough epochs
        for self.epoch in epoch_bar:
            if isinstance(self.trainloader.sampler, DS):
                self.trainloader.sampler.set_epoch(self.epoch)            
            self.model.train()             
            iter_bar = enumerate(self.trainloader)
            if self.is_rank0:
                iter_bar = tqdm(iter_bar, desc='Training', leave=False, total=len(self.trainloader))
            
            # Looping through batches
            for it, batch in iter_bar:
                self.iters += 1
                self.global_step(batch=batch, it=it)   
                
                # going through epoch step
                if self.val_every != np.inf:
                    if (self.iters % int(self.val_every * self.epoch_steps) == 0): 
                        synchronize()
                        self.epoch_step()  
                        self.model.train()         
                        
                synchronize()   
                
            if not self.save_best_model and not self.is_grid_search:
                self.best_model = model_to_CPU_state(self.feature_extractor)
                self.save_session()            
                
        print_ddp(" ==> Training done")
        if not self.is_grid_search:
            self.save_session(verbose=True)
        synchronize()
        
    def global_step(self, **kwargs):
        self.optimizer.zero_grad()
        
        # get batch
        images, labels = kwargs['batch']  
        if len(labels) == 2 and isinstance(labels, list):
            ids    = labels[1]
            labels = labels[0]

        # go through the model
        with autocast(self.use_mixed_precision):
            loss = self.model(images) 
                
        # backprop
        if not self.use_mixed_precision:
            loss.backward() 
            if self.grad_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
            self.optimizer.step()  
        else:
            self.scaler.scale(loss).backward()
            if self.grad_clipping:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
            self.scaler.step(self.optimizer)
            self.scaler.update()  
        
        if self.use_momentum:
            if ddp_is_on():
                # assuming that all models are at the stame state 
                # Otherwise this is wrong and we need to synchronise the weights first!!!!
                self.model.module.ema_update(self.iters)
            else:
                self.model.ema_update(self.iters)
            

        self.scheduler.step(self.val_acc, self.val_loss)  
        if self.iters % self.log_every == 0 or (self.iters == 1 and not self.is_grid_search):
            loss = dist_average_tensor(loss)
            if self.is_rank0:
                self.logging({'train_loss': loss.item(),
                             'learning_rate': self.get_lr()}) 
    
    def epoch_step(self, **kwargs):    
        self.evaluate()
        if not self.is_grid_search:
            self.save_session()        
     
    def evaluate(self, dataloader=None, **kwargs):
        """Validation loop function.            
        This is pretty much the same thing with global_step() but with torch.no_grad()
        Also note that DDP is not used here. There is not much point to DDP, since 
        we are not doing backprop anyway.
        """
        self.build_feature_bank()
            
        if not self.is_rank0: return
        # Note: I am removing DDP from evaluation since it is slightly slower 
        self.model.eval()
        if dataloader == None:
            dataloader=self.valloader
            
        if not len(dataloader):
            self.best_model = model_to_CPU_state(self.feature_extractor)
            self.model.train()
            return

        n_classes = dataloader.dataset.n_classes
        knn_nhood = dataloader.dataset.knn_nhood
        target_metric = dataloader.dataset.target_metric
        if self.is_rank0:
            knn_metric = self.metric_fn(n_classes, dataloader.dataset.int_to_labels, mode="knn_val")
            iter_bar = tqdm(dataloader, desc='Validating', leave=False, total=len(dataloader))
        else:
            iter_bar = dataloader
            
        self.val_loss = None
        feature_bank = []
        with torch.no_grad():
            for images, labels in iter_bar:
                if len(labels) == 2 and isinstance(labels, list):
                    ids    = labels[1]
                    labels = labels[0]
                    

                labels = labels.to(self.device_id, non_blocking=True)
                images = images.to(self.device_id, non_blocking=True)

                if is_ddp(self.model):
                    _, features = self.model.module(images, return_embedding=True)
                else:
                    _, features = self.model(images, return_embedding=True)                  

                if self.log_embeddings:
                    feature_bank.append(features.clone().detach().cpu())

                # knn_eval (always True)
                
                features = F.normalize(features, dim=1)
                pred_labels = self.knn_predict(feature = features, 
                                               feature_bank = self.feature_bank, 
                                               feature_labels =  self.targets_bank, 
                                               knn_k = knn_nhood, knn_t = 0.1, classes=n_classes,
                                               multi_label = not dataloader.dataset.is_multiclass)
                knn_metric.add_preds(pred_labels, labels, using_knn=True)

        # building Umap embeddings        
        if self.log_embeddings:
            self.build_umaps(feature_bank, dataloader, labels = knn_metric.truths, mode='val')

        eval_metrics = knn_metric.get_value(use_dist=isinstance(dataloader,DS))
        self.val_acc = eval_metrics[f"knn_val_{target_metric}"]
        
        if not self.is_grid_search:
            if self.report_intermediate_steps:
                self.logging(eval_metrics)
            if self.val_acc > self.best_val_acc:
                self.best_val_acc = self.val_acc
                if self.save_best_model:
                    self.best_model = model_to_CPU_state(self.feature_extractor)
            if not self.save_best_model:
                self.best_model = model_to_CPU_state(self.feature_extractor)
        self.model.train()

                    
    @property
    def feature_extractor(self):
        return BYOL_to_classifier(self.model)
                
def BYOL_to_classifier(net):
    if is_parallel(net):
        return net.module.online_encoder
    else:
        return net.online_encoder        
