
import torch
import torch.nn as nn
import logging
from omegaconf import OmegaConf
from functools import partial
from timesformer.models.helpers import load_pretrained
from timesformer.models.vit import VisionTransformer
from utils import make_intrinsics_layer

class ViTEncoder(nn.Module):
    def __init__(self, config):
        super(ViTEncoder, self).__init__()

        self.vit = VisionTransformer(
            img_size=config.vit.image_size,
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
            attention_type='divided_space_time'
        )

        if config.vit.pretrained:
            logging.info("Loading pretrained ViT weights")
            load_pretrained(
                self.vit,
                cfg=config.vit.default_cfg,
                num_classes=6,
                img_size=config.vit.image_size[0],
                num_frames=2,
                num_patches=(config.vit.image_size[0] // config.vit.patch_size) ** 2,
                attention_type='divided_space_time',
                pretrained_model=""
            )

        self.vit.head = nn.Identity()

    def forward(self, im0, im1):
        im0 = im0.unsqueeze(1)  # B x 1 x C x H x W
        im1 = im1.unsqueeze(1)
        x = torch.cat([im0, im1], dim=1)  # B x 2 x C x H x W
        x = x.transpose(1, 2)             # B x C x 2 x H x W
        x = self.vit(x)
        if x.dim() == 3:
            x = x.mean(dim=1)  # B x dim_emb
        return x
    

class HomographyRegressor(nn.Module):
    def __init__(self, config):
        super(HomographyRegressor, self).__init__()
        self.encoder = ViTEncoder(config)
        self.regressor = nn.Sequential(
            nn.Linear(config.vit.dim_emb, 256),
            nn.ReLU(),
            nn.Linear(256, 8)  # Homography: 8 DoF
        )

    def forward(self, data):
        x = self.encoder(data['view0']['image'], data['view1']['image'])
        return self.regressor(x)


class VORegressor(nn.Module):
    def __init__(self, config):
        super(VORegressor, self).__init__()
        self.encoder = ViTEncoder(config)

        # Load pretrained homography weights into encoder
        logging.info("Loading homography encoder weights")
        checkpoint = torch.load(config.vit.homography_weights, map_location=torch.device("cuda"))
        state_dict = checkpoint['model_state_dict']
        self.encoder.load_state_dict(state_dict, strict=False)

        self.adapter = nn.Sequential(
            nn.Conv2d(5, 3, kernel_size=1, bias=False)
        )

        with torch.no_grad():
            # Identity for RGB, zero for coordinates
            weight = torch.zeros(3, 5, 1, 1)
            weight[:, :3] = torch.eye(3).unsqueeze(-1).unsqueeze(-1)
            self.adapter[0].weight.copy_(weight)
        
        self.regressor = nn.Sequential(
            nn.Linear(config.vit.dim_emb, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

    def forward(self, data):
        il = make_intrinsics_layer(data['view0']['image'].shape[2],
                                   data['view0']['image'].shape[3],
                                   data['K'])
        
        im0 = torch.cat([data['view0']['image'], il], dim=1)
        im1 = torch.cat([data['view1']['image'], il], dim=1)
        im0 = self.adapter(im0)
        im1 = self.adapter(im1)
        x = self.encoder(im0, im1)

        return self.regressor(x)


# Example usage
if __name__ == "__main__":
    config = {
        'vit': {
            'image_size': [640, 640],
            'pretrained': False,
            'homography_weights': 'models/deep_homography_vit.pth',
            'patch_size': 16,
            'dim_emb': 384,
            'depth': 12,
            'heads': 6
        }
    }

    model = VORegressor(OmegaConf.create(config))
    
    data = {
        'view0': {'image': torch.randn(1, 3, 640, 640)},
        'view1': {'image': torch.randn(1, 3, 640, 640)},
        'K': torch.tensor([[600.0, 600.0, 320.0, 240.0]])
    }

    output = model(data)
    print(output.shape)  # torch.Size([1, 6])
