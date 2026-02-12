import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import h5py

from models.netvlad_architecture import NetVLAD
from models.pointpillar_custom import PointPillars
from transformers import ViTFeatureExtractor, ViTModel, AutoImageProcessor, AutoModel
from peft import LoraConfig, get_peft_model, TaskType

class PPDinoVlad(nn.Module):
    def __init__(self, device):
        super(PPDinoVlad, self).__init__()


        #self.local_feat_dim = 128
        #self.global_feat_dim = self.local_feat_dim*64
        
        # Loading PointPillars

        voxel_size = [0.625, 0.625, 0.1]
        pc_range = [-40,-40,-1,40,40,4]
        max_pts, max_pillars = 300, 2500
       

        self.pp = PointPillars(voxel_size, pc_range, max_pts, max_pillars).to(device)
        
        
        # Model setup - DINO
        #print("Using dinov2 Small as Encoder")
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', local_files_only=True)
        self.base_model = AutoModel.from_pretrained('facebook/dinov2-base', local_files_only=True).to(device)
  
        self.patch_size = self.base_model.config.patch_size

        '''
        #Train the last three layers only - NOT NEEDED IF USING LoRA
        for name, param in self.base_model.named_parameters():
            if 'encoder.layer.9' in name or 'encoder.layer.10' in name or 'encoder.layer.11' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        '''
        


        '''
        # DEBUG - Inspect LoRA target modules
        print("--- DINOv2 Model Parameters ---")
        for name, module in self.base_model.named_modules():
            # Only print modules that are Linear layers (or have weights)
            if isinstance(module, nn.Linear):
                print(name)
        print("-------------------------------")
        '''
        
        # Add LoRA layers to the model
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "key", "value", "dense"],  # <-- this works for DINOv3
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        



        self.base_model = get_peft_model(self.base_model, lora_config).to(device)
        print("LoRA layers added to DINOv2 backbone")
        

        # NetVLAD Projection Head
        self.projection_model = NetVLAD(num_clusters=32, dim=768).to(device)

        # Freeze the base model parameters
        for param in self.projection_model.parameters():
            param.requires_grad = True

        #Small MLP to reduce dimension from 49152 to 4096
        '''
        self.reduction_head = nn.Sequential(
            nn.Linear(64 * 768, 4096),  # From 49152 to 4096
            nn.BatchNorm1d(4096),
            nn.ReLU()
            # Optional: Add a final FC layer to 1024 if needed
        ).to(device)
        '''
        '''
        self.reduction_head = nn.Sequential(
                            nn.Linear(32 * 768, 4096),
                            nn.BatchNorm1d(4096),
                            nn.ReLU(),
                            nn.Linear(4096, 1024),
                            nn.BatchNorm1d(1024),
                            nn.ReLU()
                            ).to(device)
        '''

       
    def forward(self, x):

        imgs = self.pp(x)

        imgs_expanded = imgs.expand(-1, 3, -1, -1)
        
        feature_map = self.base_model(pixel_values=imgs_expanded).last_hidden_state[:, 1:, :]

        B, N, D = feature_map.shape
        
        h_patches = x.shape[2] // self.patch_size
        w_patches = x.shape[3] // self.patch_size
        
        imgs_embeddings = feature_map.transpose(1,2).reshape(B, D, h_patches, w_patches)

        global_desc = self.projection_model(imgs_embeddings)
        

        return imgs_embeddings, F.normalize(global_desc, p=2, dim=1)
    
