import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import h5py

from models.netvlad_architecture import NetVLAD
from transformers import ViTFeatureExtractor, ViTModel, AutoImageProcessor, AutoModel
from peft import LoraConfig, get_peft_model, TaskType

class Dino3(nn.Module):
    def __init__(self, device):
        super(Dino3, self).__init__()


        #self.local_feat_dim = 128
        #self.global_feat_dim = self.local_feat_dim*64
        

       
        # Model setup - DINO
        #print("Using dinov2 Small as Encoder")
        #processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', local_files_only=True)
        #self.base_model = AutoModel.from_pretrained('facebook/dinov2-base', local_files_only=True).to(device)
        
        access_token = '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'
        self.base_model = AutoModel.from_pretrained('facebook/dinov3-vitb16-pretrain-lvd1689m', 
                                                    token=access_token).to(device)

        '''
        #Train the last three layers only
        for name, param in self.base_model.named_parameters():
            if 'encoder.layer.9' in name or 'encoder.layer.10' in name or 'encoder.layer.11' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        '''

        self.patch_size = self.base_model.config.patch_size


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
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # <-- this works for DINOv3
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        



        self.base_model = get_peft_model(self.base_model, lora_config).to(device)
        print("LoRA layers added to DINOv3 backbone")
        


        '''
        # Compute padding needed (pad on right and bottom only)
        pad_width = (patch_size - width % patch_size) % patch_size
        pad_height = (patch_size - height % patch_size) % patch_size

        # Pad in (left, top, right, bottom) format
        padding = (int(pad_width/2), int(pad_height/2), int(pad_width/2), int(pad_height/2))

        projection_dimension = 128

        local_feat_dim = projection_dimension
        global_feat_dim = local_feat_dim*64
        '''
        # NetVLAD Projection Head
        #self.projection_model = NetVLAD(num_clusters=32, dim=768).to(device)

        # Freeze the base model parameters
        #for param in self.projection_model.parameters():
        #    param.requires_grad = True

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

        imgs_expanded = x.expand(-1, 3, -1, -1)

        outputs = self.base_model(pixel_values=imgs_expanded)
        feature_map = outputs.last_hidden_state[:, 1:, :]  # drop CLS

        B, N, D = feature_map.shape

        # True patch embed layer
        patch_embed = (
            self.base_model
                .base_model
                .model
                .embeddings
                .patch_embeddings
        )

        # Compute real patch grid
        with torch.no_grad():
            patch_tokens = patch_embed(imgs_expanded)
            _, _, H_p, W_p = patch_tokens.shape
        num_patches = H_p * W_p

        # ---- FIX: remove register tokens ----
        spatial_tokens = feature_map[:, :num_patches, :]

        # reshape using true patch grid
        imgs_embeddings = spatial_tokens.transpose(1, 2).reshape(B, D, H_p, W_p)

        #global_desc = self.projection_model(imgs_embeddings)

        # or concatenated mean + std
        mean_pool = imgs_embeddings.mean(dim=2).mean(dim=2)
        std_pool = imgs_embeddings.std(dim=2).std(dim=2)

        global_descs = torch.cat([mean_pool, std_pool], dim=1)
        global_descs = F.normalize(global_descs, p=2, dim=1)

        return imgs_embeddings, F.normalize(global_descs, p=2, dim=1)
