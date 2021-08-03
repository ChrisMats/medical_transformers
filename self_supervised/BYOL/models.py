from defaults.bases import *
from torch.cuda.amp import autocast

class BYOL_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        assert len(x) == len(y) <= 2, f"Expecting at most 2 augmented views but {len(x)} found"
        loss = []
        for x_view, y_view in zip(x, y):
            x_view = F.normalize(x_view, dim=-1, p=2)
            y_view = F.normalize(y_view, dim=-1, p=2)
            _loss =  2 - 2 * (x_view * y_view).sum(dim=-1)
            loss.append(_loss)
        return sum(loss).mean()

    
class SimSiam_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        assert len(x) == len(y) <= 2, f"Expecting at most 2 augmented views but {len(x)} found"
        loss = []
        for x_view, y_view in zip(x, y):
            _loss = - self.cos_sim(x_view, y_view).mean() / 2
            loss.append(_loss)
        return sum(loss).mean()
    
    
class Prediction_MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_size = 4096):
        super().__init__()    
        self.net = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_size))
    
    def forward(self, x):
        return self.net(x)
    
    
class BYOLHead(BaseModel):
    def __init__(self, in_size, out_size, hidden_size = 4096, num_layers=2):
        super().__init__()    
        assert 1 < num_layers < 4, "Please use 2 or 3 layers for the bottleneck module"
        
        layerlist = []
        layerlist.append(('fc_in', nn.Sequential(
                                nn.Linear(in_size, hidden_size),
                                nn.BatchNorm1d(hidden_size),
                                nn.ReLU(inplace=True))))
        if num_layers == 3:
            layerlist.append(('fc_h', nn.Sequential(
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.BatchNorm1d(hidden_size),
                                    nn.ReLU(inplace=True))))

        layerlist.append(('fc_out', nn.Sequential(
                                nn.Linear(hidden_size, out_size),
                                nn.BatchNorm1d(out_size))))
        
        self.net = nn.Sequential(OrderedDict(layerlist))
            
    def forward(self, x):
        assert isinstance(x, torch.Tensor), f"Expecting single view tensor but found {type(x)}"
        return self.net(x)     

class BYOL(BaseModel):
    def __init__(
        self,
        online_encoder, target_encoder, momentum_iters,
        bottleneck_layers = None,        
        projection_size = None,
        projection_hidden_size = None,
        prediction_hidden_size = None,
        moving_average_decay = 0.99,
        use_momentum = True
    ):
        super().__init__()
        self.use_momentum = use_momentum        
        # define loss functions and projections etc for BYOL and SimSiam
        if use_momentum:
            # Note that I use BN for the projection MLP since I got better resutls during the early steps
            self.loss_fn = BYOL_loss()
            bottleneck_layers = 2 if bottleneck_layers is None else bottleneck_layers            
            projection_size = 256 if projection_size is None else projection_size
            projection_hidden_size = 4096 if projection_hidden_size is None else projection_hidden_size
            prediction_hidden_size = 4096 if prediction_hidden_size is None else prediction_hidden_size
        else:
            self.loss_fn = SimSiam_loss()
            bottleneck_layers = 3 if bottleneck_layers is None else bottleneck_layers
            projection_size = 2048
            projection_hidden_size = 2048 if projection_hidden_size is None else projection_hidden_size
            prediction_hidden_size = 512 if prediction_hidden_size is None else prediction_hidden_size
            
        input_dim = online_encoder.fc.in_features
        self.n_classes = online_encoder.n_classes
        self.img_channels = online_encoder.img_channels        
        
        # modify online model
        self.online_encoder = online_encoder
        self.online_encoder.fc = BYOLHead(input_dim, projection_size, 
                                                projection_hidden_size, num_layers=bottleneck_layers)
        self.predictor = Prediction_MLP(projection_size, projection_size, prediction_hidden_size)
        
        # modify target model
        self.target_encoder = target_encoder
        self.target_encoder.fc = BYOLHead(input_dim, projection_size, 
                                                projection_hidden_size, num_layers=bottleneck_layers)        
        self.target_encoder.fc.load_state_dict(deepcopy(self.online_encoder.fc.state_dict()))
        assert modules_are_equal(self.online_encoder, self.target_encoder), "The Teacher and the Student must have the same initial weights"

        # freezing teacher
        self.freeze_submodel(self.target_encoder)

        # init EMA (for BYOL only)
        self.ema_updater = EMA(moving_average_decay)
        self.momentum_scheduler = CosineSchedulerWithWarmup(base_value=moving_average_decay, 
                                                            final_value=1., iters=momentum_iters)        
        
        # send the BYOL wrapped model to the original model's GPU ID
        self.to(self.device_id)

    def ema_update(self, it):
        self.ema_updater.beta = self.momentum_scheduler(it)
        for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = self.ema_updater(online_params.data, target_params.data)     

    def forward(self, x, return_embedding = False, adverserial_targets = None):
        with autocast(self.use_mixed_precision):
            if return_embedding:
                assert isinstance(x, torch.Tensor), f"Expecting single view tensor but found {type(x)}"
                x = x.to(self.device_id, non_blocking=True)
                return None, self.online_encoder.backbone(x)

            x = [x_view.to(self.device_id, non_blocking=True) for x_view in x]

            # get the 2 augmented views of the online net
            x_online, x_emb = zip(*[self.online_encoder(x_view, return_embedding=True) for x_view in x])
            x_online = [self.predictor(x_view) for x_view in x_online]

            with torch.no_grad():
                # get the 2 augmented views of the target net
                target_encoder = self.target_encoder if self.use_momentum else self.online_encoder
                x_target = [target_encoder(x_view) for x_view in x]

            return self.loss_fn(x_online, x_target)
    