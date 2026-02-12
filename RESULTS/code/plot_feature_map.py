import os
import random
import time
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from os.path import isfile, join
from PIL import Image
from sklearn.decomposition import PCA

import dataset_manager.kitti_dataset as kitti_dataset
import dataset_manager.nclt_dataset as nclt_dataset
import dataset_manager.helilpr_dataset as helilpr_dataset
import dataset_manager.toyota_dataset as toyota_dataset
from models.model_selector import model_selector

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend

def load_weights(path, model):
    ckpt_path = os.path.join(path, 'model_best.pth.tar')
    if not os.path.isfile(ckpt_path):
        print(f"Warning: checkpoint not found at {ckpt_path}")
        return model
    print(f"Loading weights from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    return model

if __name__ == "__main__":

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    representation='polar_multi'
    
    MODEL = "dino3vlad"
    # proj_dim is not used here, only the model instance is needed
    model, _ = model_selector(MODEL, device) 
    model = load_weights(f"./representation_analysis/RESULTS/Trainings/{representation}/checkpoint/", model)

    model.eval()
    model.to(device)

    # 1. Load the input image
    im_path = "/dataset/kitti-dataset/lidar_representations/representation_analysis/sequences/00/"+ representation +"/000000.png"
    #im_path = "/dataset/helilpr/lidar_representations/representation_analysis/rb01/"+representation+"/000100.png"
    #im_path = "/dataset/nclt-dataset/lidar_representations/representation_analysis/2012-02-04/"+representation+"/000000.png"
    im_gray = cv2.imread(im_path, 0)
    
    if im_gray is None:
        print(f"Error: Could not load image from {im_path}")
        exit()

    # Preprocessing for the model:
    img_float = (im_gray.astype(np.float32)) / 256.0 # Normalize to [0, 1]
    
    # Convert original image to 3 channels for overlaying
    img_original_rgb = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)
    img_original_rgb = img_original_rgb.astype(np.float32) / 255.0 # Normalize to [0, 1]

    img_rgb_input_tensor = img_float[np.newaxis, :, :].repeat(3, 0) # 1xHxW -> 3xHxW
    img_batch = img_rgb_input_tensor[np.newaxis, :, :, :] # 1x3xHxW (batch)
    
    img_tensor = torch.from_numpy(img_batch).to(device)

    # --- Extract local_feat from your model ---
    with torch.no_grad():
        local_feat, global_desc = model(img_tensor)

    # local_feat: [1, C, Hf, Wf]
    C, Hf, Wf = local_feat.shape[1:]

    # --- PCA FEATURE VISUALIZATION (Improved Version) ---

from sklearn.preprocessing import StandardScaler
import cv2
import numpy as np

# 1. Flatten features
hmap = local_feat.squeeze(0).cpu().numpy()     # [C, Hf, Wf]
feat = np.moveaxis(hmap, 0, -1)                # [Hf, Wf, C]
flat = feat.reshape(-1, C).astype(np.float32)  # [N, C]

# 2. Standardize BEFORE PCA (prevents high-variance channels dominating)
flat_std = StandardScaler().fit_transform(flat)

# 3. PCA → 3 components
pca = PCA(n_components=3)
flat_pca = pca.fit_transform(flat_std)  # [N, 3]

# 4. Global contrast normalization (NOT per-channel)
flat_pca /= (np.abs(flat_pca).max() + 1e-9)
feat_pca = flat_pca.reshape(feat.shape[0], feat.shape[1], 3)

# 5. Optional: smooth feature map (helps if Hf,Wf small)
feat_smoothed = cv2.GaussianBlur(feat_pca, (0, 0), sigmaX=0.7)

# 6. Upscale using bilinear interpolation (less artifact-prone)
H_in, W_in = im_gray.shape
#feat_up = cv2.resize(feat_smoothed, (W_in, H_in), interpolation=cv2.INTER_LINEAR)
feat_up = cv2.resize(feat_smoothed, (128, 128), interpolation=cv2.INTER_LINEAR)
# 7. Normalize into [0,1]
feat_up = (feat_up - feat_up.min()) / (feat_up.max() - feat_up.min() + 1e-9)
img_original_rgb = cv2.resize(img_original_rgb, (128, 128), interpolation=cv2.INTER_LINEAR)

# 8. Overlay
overlay = cv2.addWeighted(img_original_rgb, 0.6, feat_up, 0.4, 0)

cv2.imwrite("feature_map_pca_overlay.png", (overlay * 255).astype(np.uint8))
