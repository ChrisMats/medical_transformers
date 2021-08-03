import wandb
import defaults
from defaults.bases import *
import matplotlib.pyplot as plt
from .wrappers import DefaultWrapper, dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler as DS

        
class Trainer(BaseTrainer):
    """Main trainer class.

    Initializes with a DefaultWrapper instance as its input. 
    Call trainer.train() to train and validate or call trainer.test()
    Training with standard DDP: a model is trainedon multiple machines/gpus using distributed gradients. 
    """
    def __init__(self, wraped_defs):
        """Initialize the trainer instance.
        
        This function clones its attributes from the DefaultWrapper instance or generates
        them from the .json file. 
        """
        super().__init__()

        self.parameters = wraped_defs.parameters
        self.is_supervised = wraped_defs.is_supervised        
        self.training_params = self.parameters.training_params
        self.attr_from_dict(self.training_params)
        self.attr_from_dict(wraped_defs.dataloaders)
        self.epoch_steps = len(self.trainloader)
        self.total_steps = int(len(self.trainloader) * self.epochs)
        
        self.model = wraped_defs.model
        self.criterion = wraped_defs.criterion        
        self.optimizer = wraped_defs.optimizer 
        self.scheduler = wraped_defs.schedulers
        self.metric_fn = wraped_defs.metric
        
        self.org_model_state = model_to_CPU_state(self.model)
        self.org_optimizer_state = opimizer_to_CPU_state(self.optimizer)
        self.total_step = len(self.trainloader)        
        self.best_model = deepcopy(self.org_model_state)  
        
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            self.set_models_precision(self.use_mixed_precision)        
        
        
    def train(self):
        """Main training loop."""
        self.test_mode = False
        if not self.is_grid_search:
            self.load_session(self.restore_only_model)
        self.print_train_init()
        
        n_classes = self.trainloader.dataset.n_classes
        metric = self.metric_fn(n_classes, self.trainloader.dataset.int_to_labels, mode="train")
        epoch_bar = range(self.epoch0 + 1, self.epoch0 + self.epochs + 1)
        if self.is_rank0:
            epoch_bar = tqdm(epoch_bar, desc='Epoch', leave=False)
            
        for self.epoch in epoch_bar:            
            self.model.train() 
            if isinstance(self.trainloader.sampler, DS):
                self.trainloader.sampler.set_epoch(self.epoch)            
            
            iter_bar = enumerate(self.trainloader)
            if self.is_rank0:
                iter_bar = tqdm(iter_bar, desc='Training', leave=False, total=len(self.trainloader))
            for it, batch in iter_bar:
                self.iters += 1
                self.global_step(batch=batch, metric=metric, it=it)   
                
                if self.val_every != np.inf:
                    if self.iters % int(self.val_every * self.epoch_steps) == 0: 
                        synchronize()
                        self.epoch_step()  
                        self.model.train()
                synchronize()
                
            if not self.save_best_model and not self.is_grid_search:
                self.best_model = model_to_CPU_state(self.model)   
                self.save_session()                         
                
        if self.is_rank0:         
            print(" ==> Training done")
        if not self.is_grid_search:
            self.evaluate()
            self.save_session(verbose=True)
        synchronize()
        
    def global_step(self, **kwargs):
        """Function for the standard forward/backward/update.
        
        If using DDP, metrics (e.g. accuracy) are calculated with dist.all_gather
        """
        self.optimizer.zero_grad()
        
        metric = kwargs['metric']        
        images, labels = kwargs['batch']
        if len(labels) == 2 and isinstance(labels, list):
            ids    = labels[1]
            labels = labels[0]
        labels = labels.to(self.device_id, non_blocking=True)
        images = images.to(self.device_id, non_blocking=True) 
        
        with autocast(self.use_mixed_precision):
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

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
        
        metric.add_preds(outputs, labels) # distributed gather inside
        self.scheduler.step(self.val_acc, self.val_loss)

        if not self.is_grid_search:
            if self.iters % self.log_every == 0 or self.iters == 1:
                loss = dist_average_tensor(loss)
                if self.is_rank0:
                    self.logging({'train_loss': loss.item(),
                                 'learning_rate': self.get_lr()})
                    self.logging(metric.get_value())     
                    metric.reset()                
    
    def epoch_step(self, **kwargs): 
        """Function for periodic validation, LR updates and model saving.
        
        Note that in the 2nd phase of training, the behavior is different, each model on
        each GPU is saved separately.
        """
        self.evaluate()        
        if not self.is_grid_search:
            self.save_session()                  
    
    def evaluate(self, dataloader=None, **kwargs):
        """Validation loop function.
        
        This is pretty much the same thing with global_step() but with torch.no_grad()
        Also note that DDP is not used here. There is not much point to DDP, since 
        we are not doing backprop anyway.
        """
        
        # if we are using knn evaluations build a feature bank of the train set
        if self.knn_eval:
            self.build_feature_bank()
            
        if not self.is_rank0: return
        # Note: I am removing DDP from evaluation since it is slightly slower 
        self.model.eval()
        if dataloader == None:
            dataloader=self.valloader

        if not len(dataloader):
            self.best_model = model_to_CPU_state(self.model)
            self.model.train()
            return
        
        knn_nhood = dataloader.dataset.knn_nhood
        n_classes = dataloader.dataset.n_classes
        target_metric = dataloader.dataset.target_metric
        if self.is_rank0:
            metric = self.metric_fn(n_classes, dataloader.dataset.int_to_labels, mode="val")
            if self.knn_eval:
                knn_metric = self.metric_fn(n_classes, dataloader.dataset.int_to_labels, mode="knn_val")
            iter_bar = tqdm(dataloader, desc='Validating', leave=False, total=len(dataloader))
        else:
            iter_bar = dataloader

        val_loss = []
        feature_bank = []
        with torch.no_grad():
            for images, labels in iter_bar:
                if len(labels) == 2 and isinstance(labels, list):
                    ids    = labels[1]
                    labels = labels[0]
                labels = labels.to(self.device_id, non_blocking=True)
                images = images.to(self.device_id, non_blocking=True)

                if is_ddp(self.model):
                    outputs, features = self.model.module(images, return_embedding=True)
                else:
                    outputs, features = self.model(images, return_embedding=True)
                    
                if self.log_embeddings:
                    feature_bank.append(features.clone().detach().cpu())   
                        
                if self.knn_eval:
                    features = F.normalize(features, dim=1)
                    pred_labels = self.knn_predict(feature = features, 
                                                   feature_bank=self.feature_bank, 
                                                   feature_labels= self.targets_bank, 
                                                   knn_k=knn_nhood, knn_t=0.1, classes=n_classes, 
                                                   multi_label = not dataloader.dataset.is_multiclass)
                    knn_metric.add_preds(pred_labels, labels, using_knn=True)

                loss = self.criterion(outputs, labels)
                val_loss.append(loss.item())
                metric.add_preds(outputs, labels)
                
        # building Umap embeddings
        if self.log_embeddings:
            self.build_umaps(feature_bank, dataloader, labels = knn_metric.truths, mode='val')
            
        self.val_loss = np.array(val_loss).mean()
        eval_metrics = metric.get_value(use_dist=isinstance(dataloader,DS))
        if self.knn_eval:
            eval_metrics.update(knn_metric.get_value(use_dist=isinstance(dataloader,DS)))
        self.val_acc = eval_metrics[f"val_{target_metric}"]

        if not self.is_grid_search:
            if self.report_intermediate_steps:
                self.logging(eval_metrics)
                self.logging({'val_loss': round(self.val_loss, 5)})
            if self.val_acc > self.best_val_acc:
                self.best_val_acc = self.val_acc
                if self.save_best_model:
                    self.best_model = model_to_CPU_state(self.model)
            if self.val_loss <= self.best_val_loss:
                self.best_val_loss = self.val_loss
            if not self.save_best_model:
                self.best_model = model_to_CPU_state(self.model)
        self.model.train()
        
    def test(self, dataloader=None, **kwargs):
        """Test function.
        
        Just be careful you are not explicitly passing the wrong dataset here.
        Otherwise it will use the test set.
        """
        if not self.is_rank0: return
        if self.log_embeddings:
            self.build_feature_bank()
            
        self.test_mode = True
        self.restore_session = True
        self.restore_only_model = True
        self.set_models_precision(False)
        try:
            self.load_session(self.restore_only_model)
        except:
            print("Full checkpoint not found... Proceeding with partial model (assuming transfer learning is ON)")
        self.model.eval()
        if dataloader == None:
            dataloader=self.testloader  
            
        results_dir = os.path.join(self.save_dir, 'results', self.model_name)
        metrics_path = os.path.join(results_dir, "metrics_results.json")
        check_dir(results_dir)     

        test_loss = []
        feature_bank = []
        results = edict()
        knn_nhood = dataloader.dataset.knn_nhood
        n_classes = dataloader.dataset.n_classes    
        target_metric = dataloader.dataset.target_metric
        if self.is_supervised:
            metric = self.metric_fn(n_classes, dataloader.dataset.int_to_labels, mode="test")
        if self.knn_eval or not self.is_supervised:
            knn_metric = self.metric_fn(n_classes, dataloader.dataset.int_to_labels, mode="knn_val")        
        iter_bar = tqdm(dataloader, desc='Testing', leave=True, total=len(dataloader))
        
        with torch.no_grad():
            for images, labels in iter_bar: 
                if len(labels) == 2 and isinstance(labels, list):
                    ids    = labels[1]
                    labels = labels[0]
                labels = labels.to(self.device_id, non_blocking=True)
                images = images.to(self.device_id, non_blocking=True)                   
                
                if is_ddp(self.model):
                    outputs, features = self.model.module(images, return_embedding=True)
                else:
                    outputs, features = self.model(images, return_embedding=True)
                    
                if self.log_embeddings:
                    feature_bank.append(features.clone().detach().cpu())                      
                if self.knn_eval:
                    features = F.normalize(features, dim=1)
                    pred_labels = self.knn_predict(feature = features, 
                                                   feature_bank=self.feature_bank, 
                                                   feature_labels= self.targets_bank, 
                                                   knn_k=knn_nhood, knn_t=0.1, classes=n_classes,
                                                   multi_label = not dataloader.dataset.is_multiclass)
                    knn_metric.add_preds(pred_labels, labels, using_knn=True) 
                if self.is_supervised:
                    loss = self.criterion(outputs, labels)
                    test_loss.append(loss.item())
                    metric.add_preds(outputs, labels)
                
        if self.log_embeddings:
            build_umaps(feature_bank, dataloader, labels = metric.truths if self.is_supervised else knn_metric.truths, 
                        id_bank = id_bank, mode = 'test', wandb_logging=False)                         
        
        self.test_loss = np.array(test_loss).mean() if test_loss else None
        test_metrics = {}
        if self.is_supervised:
            test_metrics = metric.get_value(use_dist=isinstance(dataloader,DS))
        if self.knn_eval or not self.is_supervised:
            test_metrics.update(knn_metric.get_value(use_dist=isinstance(dataloader,DS)))  
        if self.is_supervised:
            self.test_acc = test_metrics[f"test_{target_metric}"]
            test_metrics['test_loss'] = round(self.test_loss, 5)
        else:
            self.test_acc = test_metrics.knn_test_accuracy
        
        self.model.train()
        self.set_models_precision(self.use_mixed_precision)
        save_json(test_metrics, metrics_path)
        
        print('\n',"--"*5, "{} evaluated on the test set".format(self.model_name), "--"*5,'\n')
        test_metrics = pd.DataFrame.from_dict(test_metrics, orient='index').T
        print(tabulate(test_metrics, headers = 'keys', tablefmt = 'psql'))
        print('\n',"--"*35, '\n')        

        
    def build_umaps(self, feature_bank, dataloader, labels = None, mode='', wandb_logging=True):
        if not dataloader.dataset.is_multiclass: return
        feature_bank = torch.cat(feature_bank, dim=0).numpy()
        umap_path = self.get_embedding_path(mode=mode, iters=self.iters)
        create_umap_embeddings(feature_bank, labels, 
                                   label_mapper=dataloader.dataset.int_to_labels,
                                   umap_path=umap_path)

        if wandb_logging:  
            if self.use_tensorboard:
                umap_plot = plt.imread(umap_path)
                self.logging({"umap_embeddings": [umap_plot[:,:,:3]]})
            else:
                umap_plot = Image.open(umap_path) 
                self.logging({"umap_embeddings": [wandb.Image(umap_plot, 
                                                          caption=self.model_name)]})
            
    def build_feature_bank(self, dataloader=None, **kwargs):
        """Build feature bank function.
        
        This function is meant to store the feature representation of the training images along with their respective labels 

        This is pretty much the same thing with global_step() but with torch.no_grad()
        Also note that DDP is not used here. There is not much point to DDP, since 
        we are not doing backprop anyway.
        """
        
        self.model.eval()
        if dataloader is None:
            dataloader = self.fbank_loader         
        
        n_classes = dataloader.dataset.n_classes
        if self.is_rank0:
            iter_bar = tqdm(dataloader, desc='Building Feature Bank', leave=False, total=len(dataloader))
        else:
            iter_bar = dataloader
            
        with torch.no_grad():
            
            self.feature_bank = []
            self.targets_bank = []   

            for images, labels in iter_bar:
                if len(labels) == 2 and isinstance(labels, list):
                    ids    = labels[1]
                    labels = labels[0]
                labels = labels.to(self.device_id, non_blocking=True)
                images = images.to(self.device_id, non_blocking=True)                   
                
                if is_ddp(self.model):
                    _, feature = self.model.module(images, return_embedding=True)
                else:
                    _, feature = self.model(images, return_embedding=True)
                  
                feature = F.normalize(feature, dim=1)
                self.feature_bank.append(feature)
                self.targets_bank.append(labels)

            self.feature_bank = torch.cat(self.feature_bank, dim=0).t().contiguous()
            self.targets_bank = torch.cat(self.targets_bank, dim=0).t().contiguous()

            synchronize()
            self.feature_bank = dist_gather(self.feature_bank, cat_dim=-1)
            self.targets_bank = dist_gather(self.targets_bank, cat_dim=-1)
        self.model.train()
        
    def lr_grid_search(self, min_pow=-5, max_pow=-1, resolution=20, n_epochs=5, 
                       random_lr=False, report_intermediate_steps=False, keep_schedule=False):
        """Hyperparameter search function.
        
        Since we are using well-known datasets this is not necessary, but here for completeness.
        """
        res_dir = "grid_search_results"
        res_dir = os.path.join(self.save_dir, res_dir)        
        self.is_grid_search = True
        self.save_best_model = False
        self.epochs = n_epochs
        if not keep_schedule:
            self.scheduler =  MixedLRScheduler([None], [None], None) 
        pref_m = self.model_name
        self.model_name = 'grid_search'
        self.save_every = float("inf")   
        self.report_intermediate_steps = report_intermediate_steps
        if self.report_intermediate_steps:
            self.val_every = 1            
        else:
            self.log_every = float("inf")
            self.val_every = float("inf")
        
        v_losses = []
        v_accs = []
        if not random_lr:
            e = np.linspace(min_pow, max_pow, resolution)
            lr_points = 10**(e)
        else:
            lr_points = np.random.uniform(min_pow, max_pow, resolution)
            lr_points = 10**(e)
                    
        check_dir(res_dir)
        out_name = pref_m + "_grid_search_out.txt"
        out_name = os.path.join(res_dir, out_name)
        with open(out_name, "w") as text_file:
            print('learning rate \t val_loss \t val_AUC', file=text_file)
        for lr in tqdm(lr_points, desc='Grid search cycles', leave=False):
            if report_intermediate_steps:
                wandb.init(project=pref_m + '_grid_search', name=str(lr), reinit=True)     
        
            self.optimizer.param_groups[0]['lr'] = lr
            if keep_schedule:
                raise NotImplementedError
            self.train()
            self.evaluate()
            v_losses.append(self.val_loss)
            v_accs.append(self.val_acc)
            if report_intermediate_steps:
                self.logging({'val_loss': self.val_loss,
                              'val_acc': self.val_acc})  
            with open(out_name, "a") as text_file:
                print('{} \t {} \t {}'.format(lr,self.val_loss,self.val_acc), file=text_file)
            self.reset()
            self.val_loss = float("inf")
            self.best_val_loss = float("inf")
            self.val_acc = 0.
            self.best_val_acc = 0.
            self.iters = 0
            self.epoch0 = 0
            self.epoch = 0 
            if report_intermediate_steps:
                wandb.uninit()
            
        arg_best_acc = np.argmax(v_accs)
        best_acc = v_accs[arg_best_acc]
        best_lr_acc = lr_points[arg_best_acc]

        arg_best_vloss = np.argmin(v_losses)
        best_vloss = v_losses[arg_best_vloss]
        best_lr_vloss = lr_points[arg_best_vloss]

        print("The best val_acc is {} for lr = {}".format(best_acc, best_lr_acc))
        print("The best val_loss is {} for lr = {}".format(best_vloss, best_lr_vloss))
        
        fig, axs = plt.subplots(1,2, figsize=(15, 6))
        axs = axs.ravel()
        fig.suptitle('Grid search results')
        axs[0].plot(lr_points, v_losses)
        axs[0].scatter(best_lr_vloss, best_vloss, marker='*', c='r', s=100)
        axs[0].plot([best_lr_vloss]*2, [0, best_vloss], linestyle='--', c='r', alpha=0.5)
        axs[0].plot([lr_points[0], best_lr_vloss], [best_vloss]*2, linestyle='--', c='r', alpha=0.5)
        axs[0].set_xlabel('Learning rate')
        axs[0].set_ylabel('Validation loss')
        axs[0].set_xscale('log')
        axs[1].plot(lr_points, v_accs)
        axs[1].scatter(best_lr_acc, best_acc, marker='*', c='r', s=100)
        axs[1].plot([best_lr_acc]*2, [0, best_acc], linestyle='--', c='r', alpha=0.5)
        axs[1].plot([lr_points[0], best_lr_acc], [best_acc]*2, linestyle='--', c='r', alpha=0.5)
        axs[1].set_xlabel('Learning rate')
        axs[1].set_ylabel('Validation acc')
        axs[1].set_xscale('log')
        plt.savefig(os.path.join(res_dir, pref_m + '_grid_search_out.png'))   
        
    # FROM: https://github.com/leftthomas/SimCLR/blob/cee178b6cab1efab67f8b527a0cff91c9e793f5c/main.py#L48
    def knn_predict(self, feature, feature_bank, feature_labels, 
                    knn_k: int, knn_t: float, classes: int = 10, multi_label = False):
        """Helper method to run kNN predictions on features based on a feature bank

        Args:
            feature: Tensor of shape [N, D] consisting of N D-dimensional features
        feature_bank: Tensor of a database of features used for kNN
        feature_labels: Labels for the features in our feature_bank
        classes: Number of classes (e.g. 10 for CIFAR-10)
        knn_k: Number of k neighbors used for kNN
        knn_t:

        """
        
        # CHANGE TO FLAG
        if multi_label:
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            # Find the similarities between the batch samples and the feature bank
            sim_matrix = torch.mm(feature, feature_bank)


            # identify the knn_k most similar samples in the feature bank for each of the batch samples
            sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)


            # Expand the feature labels to a have a copy per batch sample
            expanded_labels = feature_labels.expand((feature.size(0),feature_labels.size(0),feature_labels.size(1)))

            # Unsqueeze and expand the similarity indicies and weights 
            sim_indices = sim_indices.unsqueeze_(1)
            sim_weight  = sim_weight.unsqueeze_(1)
            sim_indices_expanded = sim_indices.expand((sim_indices.size(0),feature_labels.size(0),sim_indices.size(2)))
            sim_weight_expanded  =  sim_weight.expand((sim_weight.size(0) ,feature_labels.size(0), sim_weight.size(2)))

            # Gather the labels of the most similar samples in the feature bank
            gathered = torch.gather(expanded_labels, dim=-1, index=sim_indices_expanded)

            # Scale the weights of the most similar samples
            sim_weight_expanded = (sim_weight_expanded / knn_t).exp()

            # weight each of the labels 
            weighted_labels = F.normalize(sim_weight_expanded,p=1,dim=2)*gathered
            pred_labels = weighted_labels.sum(axis=2)
            
            return pred_labels
        
        else:
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)

            # we do a reweighting of the similarities
            sim_weight = (sim_weight / knn_t).exp()

            # counts for each class
            one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)[:, 0]
            
        return pred_labels
    
