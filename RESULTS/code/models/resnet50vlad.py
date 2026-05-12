import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from models.netvlad_architecture import NetVLAD


class ResNet50Vlad(nn.Module):
    def __init__(self, device):
        super(ResNet50Vlad, self).__init__()

        # ============================================================
        # ResNet50 backbone
        # ============================================================

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Remove avgpool + fc
        self.base_model = nn.Sequential(*list(backbone.children())[:-2]).to(device)

        # Output channels from ResNet50 final conv block
        backbone_dim = 2048

        # ============================================================
        # NetVLAD head
        # ============================================================

        self.projection_model = NetVLAD(
            num_clusters=32,
            dim=backbone_dim
        ).to(device)

        # Train backbone + NetVLAD
        for param in self.base_model.parameters():
            param.requires_grad = True

        for param in self.projection_model.parameters():
            param.requires_grad = True

    def forward(self, x):

        # ------------------------------------------------------------
        # Input handling
        # ------------------------------------------------------------
        # If grayscale: [B,1,H,W] -> [B,3,H,W]
        # If already RGB, keep as-is
        # ------------------------------------------------------------

        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)

        # ============================================================
        # CNN feature extraction
        # ============================================================

        imgs_embeddings = self.base_model(x)
        # Shape:
        # [B, 2048, H/32, W/32]

        # ============================================================
        # NetVLAD aggregation
        # ============================================================

        global_desc = self.projection_model(imgs_embeddings)

        # L2 normalize descriptor
        global_desc = F.normalize(global_desc, p=2, dim=1)

        return imgs_embeddings, global_desc