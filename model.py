
import torch
import torch.nn as nn
import logging
from omegaconf import OmegaConf
from functools import partial
from timesformer.models.helpers import load_pretrained
from timesformer.models.vit import VisionTransformer

class HomographyRegressor(nn.Module):
    def __init__(self, config):
        super(HomographyRegressor, self).__init__()
        
        self.vit = VisionTransformer(img_size=config.vit.image_size,
                            num_classes=6,
                            patch_size=config.vit.patch_size,
                            embed_dim=config.vit.dim_emb,
                            depth=config.vit.depth,
                            num_heads=config.vit.heads,
                            mlp_ratio=4,
                            qkv_bias=True,
                            norm_layer=partial(nn.LayerNorm, eps=1e-6),
                            drop_rate=0.,
                            attn_drop_rate=0.,
                            drop_path_rate=0.1,
                            num_frames=2,
                            attention_type='divided_space_time')
        if(config.vit.pretrained):
            logging.info("Loading pretrained weights")
            load_pretrained(self.vit, cfg=config.vit.default_cfg,
                        num_classes=6, img_size=config.vit.image_size[0], num_frames=2,
                        num_patches=(config.vit.image_size[0] // config.vit.patch_size) ** 2,
                        attention_type='divided_space_time',
                        pretrained_model="")
        
        self.vit.head = nn.Identity()
        self.regressor = nn.Sequential(nn.Linear(384, 256),  
                                      nn.ReLU(),
                                      nn.Linear(256, 8))
        
    def forward(self, data):
        im0 = data['view0']['image']
        im1 = data['view1']['image']
        im0 = im0.unsqueeze(1)
        im1 = im1.unsqueeze(1)
        x = torch.cat([im0, im1], dim=1)
        x = x.transpose(1, 2)
        x = self.vit(x)
        if x.dim() == 3:  # (B, num_tokens, dim)
            x = x.mean(dim=1)
        return self.regressor(x)
               

if __name__ == "__main__":
    # Example usage
    config = {
        'vit': {
            'image_size': [640,640],
            'pretrained': False,
            'patch_size': 16,
            'dim_emb': 384,
            'depth': 12,
            'heads': 6
        }
    }
    
    model = HomographyRegressor(OmegaConf.create(config))
    
    data = {
        'view0': {
            'image': torch.randn(1, 3, 640, 640),
        },
        'view1': {
            'image': torch.randn(1, 3, 640, 640),
        }
    }
    output = model(data)
    print(output.shape)  # Should be (1, 3, 3)