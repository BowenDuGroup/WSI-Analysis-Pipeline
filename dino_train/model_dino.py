# dino models

import torch
import torch.nn as nn
import timm

def get_backbone(model_name='vit_small_patch16_224'):
    model = timm.create_model(
        model_name, 
        pretrained=False, 
        num_classes=0,
        dynamic_img_size=True 
    )
    return model

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn: layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn: layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m.bias, torch.Tensor):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        if isinstance(x, list):
            dims = torch.tensor([inp.shape[-1] for inp in x])
            idx_crops = torch.cumsum(torch.unique_consecutive(dims, return_counts=True)[1], 0)
            
            start_idx = 0
            for end_idx in idx_crops:
                _out = self.backbone(torch.cat(x[start_idx: end_idx]))
                if start_idx == 0: output = _out
                else: output = torch.cat((output, _out))
                start_idx = end_idx
            return self.head(output)
        else:
            return self.head(self.backbone(x))
