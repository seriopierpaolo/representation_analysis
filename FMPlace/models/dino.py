import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import h5py

from models.netvlad_architecture import NetVLAD
from transformers import ViTFeatureExtractor, ViTModel, AutoImageProcessor, AutoModel


class Dino(nn.Module):
    def __init__(self, device):
        super(Dino, self).__init__()


        self.local_feat_dim = 128
        self.global_feat_dim = self.local_feat_dim*64
        #self.padding = padding
        # Model setup - DINO
        print("Using dinov2 Small as Encoder")
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base') # to use local data only: local_files_only=True
        self.base_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device) # to use local data only: local_files_only=True

        #Train the last two layers only
        for name, param in self.base_model.named_parameters():
            if 'encoder.layer.9' in name or 'encoder.layer.10' in name or 'encoder.layer.11' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Get original dimensions
        #width, height = image_size
        patch_size = 36

        # Compute padding needed (pad on right and bottom only)
        #pad_width = (patch_size - width % patch_size) % patch_size
        #pad_height = (patch_size - height % patch_size) % patch_size

        # Pad in (left, top, right, bottom) format
        #padding = (int(pad_width/2), int(pad_height/2), int(pad_width/2), int(pad_height/2))

        # NetVLAD Projection Head
        self.projection_model = NetVLAD(num_clusters=64, dim=768).to(device)

        # Freeze the base model parameters
        for param in self.projection_model.parameters():
            param.requires_grad = False

        #Small MLP to reduce dimension from 49152 to 4096
        '''
        self.reduction_head = nn.Sequential(
            nn.Linear(64 * 768, 4096),  # From 49152 to 4096
            nn.BatchNorm1d(4096),
            nn.ReLU()
            # Optional: Add a final FC layer to 1024 if needed
        ).to(device)
        '''

    def forward(self, x):

        # Apply padding to make dimensions compatible with patch size
        #imgs = torch.stack([F.pad(x_subs, self.padding, value=0) for x_sub in x])


    
        # BEV point clouds have 1 channel, but DINOv2 expects 3 channels
        # Expand from [batch_size, 1, H, W] to [batch_size, 3, H, W]
        imgs_expanded = x.expand(-1, 3, -1, -1)

    
        # Get embeddings from the base model

        tokens = self.base_model(imgs_expanded).last_hidden_state[:, 1:, :]
        B, N, D = tokens.shape
        side = int(N ** 0.5)
        dino_feats = tokens.transpose(1,2).reshape(B, D, side, side)

        # pool DINO features yourself, e.g., global average:
        global_descs = dino_feats.mean(dim=2).mean(dim=2)  # if shape [B, C, H, W]
        global_descs = F.normalize(global_descs, p=2, dim=1)
        # global max pooling
        global_descs = dino_feats.max(dim=2)[0].max(dim=2)[0]
        global_descs = F.normalize(global_descs, p=2, dim=1)

        # or concatenated mean + std
        mean_pool = dino_feats.mean(dim=2).mean(dim=2)
        std_pool = dino_feats.std(dim=2).std(dim=2)
        global_descs = torch.cat([mean_pool, std_pool], dim=1)
        global_descs = F.normalize(global_descs, p=2, dim=1)


        return dino_feats, global_descs
