from .bases import *
from . import transformers
from torch.cuda.amp import autocast


class Identity(nn.Module):
    """An identity function."""
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
    
class Classifier(BaseModel):
    """A wrapper class that provides different CNN backbones.
    
    Is not intended to be used standalone. Called using the DefaultWrapper class.
    """
    def __init__(self, model_params):
        super().__init__()
        self.attr_from_dict(model_params)           
        
        if hasattr(transformers, self.backbone_type):  
            self.backbone = transformers.__dict__[self.backbone_type](**self.transformers_params, 
                                                                      pretrained=self.pretrained)
            fc_in_channels = self.backbone.embed_dim

        elif hasattr(cnn_models, self.backbone_type):
            self.backbone = cnn_models.__dict__[self.backbone_type](pretrained=self.pretrained)
            fc_in_channels = self.backbone.fc.in_features
        else:
            raise NotImplementedError                
        self.backbone.fc = Identity()                
        
        # modify stem and last layer
        self.fc = nn.Linear(fc_in_channels, self.n_classes)
        self.modify_first_layer(self.img_channels, self.pretrained)            
        
        if self.freeze_backbone:
            self.freeze_submodel(self.backbone)   

    def forward(self, x, return_embedding=False):
        with autocast(self.use_mixed_precision):
            
            if isinstance(x, list) and hasattr(cnn_models, self.backbone_type):
                idx_crops = torch.cumsum(torch.unique_consecutive(
                    torch.tensor([inp.shape[-1] for inp in x]),
                    return_counts=True,
                )[1], 0)
                start_idx = 0
                for end_idx in idx_crops:
                    _out = self.backbone(torch.cat(x[start_idx: end_idx]))
                    if start_idx == 0:
                        x_emb = _out
                    else:
                        x_emb = torch.cat((x_emb, _out))
                    start_idx = end_idx             
            else:
                x_emb = self.backbone(x)
                
            x = self.fc(x_emb)
            
            if return_embedding:
                return x, x_emb        
            else:
                return x
        
    def modify_first_layer(self, img_channels, pretrained):
        backbone_type = self.backbone.__class__.__name__
        if img_channels == 3:
            return
        if backbone_type == 'ResNet':
            conv_attrs = ['out_channels', 'kernel_size', 'stride', 
                          'padding', 'dilation', "groups", "bias", "padding_mode"]
            conv1_defs = {attr: getattr(self.backbone.conv1, attr) for attr in conv_attrs}

            pretrained_weight = self.backbone.conv1.weight.data
            pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]

            self.backbone.conv1 = nn.Conv2d(img_channels, **conv1_defs)
            if pretrained:
                self.backbone.conv1.weight.data = pretrained_weight 
                
        elif backbone_type == 'VisionTransformer':
            patch_embed_attrs = ["img_size", "patch_size"]
            patch_defs = {attr: getattr(self.backbone.patch_embed, attr) for attr in patch_embed_attrs}
            patch_defs["embed_dim"] = self.backbone.embed_dim

            pretrained_weight = self.backbone.patch_embed.proj.weight.data
            if self.backbone.patch_embed.proj.bias is not None:
                pretrained_bias = self.backbone.patch_embed.proj.bias.data
            pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]
            
            self.backbone.patch_embed = transformers.PatchEmbed(in_chans=img_channels, **patch_defs)
            if pretrained:
                self.backbone.patch_embed.proj.weight.data = pretrained_weight 
                if self.backbone.patch_embed.proj.bias is not None:
                    self.backbone.patch_embed.proj.bias.data = pretrained_bias                  
        
        else:
            raise NotImplementedError("channel modification is not implemented for {}".format(backbone_type))
