import torch
import torch.nn.functional as F

# Ensure data is normalized before input
def normalize_tensor(tensor):
    """Normalize tensor to [0, 1] range"""
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)


def get_embeddings_resnet(anchors, positives, negatives, base_model, device):
        
    # Normalize each sample in the batch
    anchors = torch.stack([normalize_tensor(x) for x in anchors])
    positives = torch.stack([normalize_tensor(x) for x in positives])
    negatives = torch.stack([normalize_tensor(x) for x in negatives])
    
    # Move to device
    anchors = anchors.to(device)
    positives = positives.to(device)
    negatives = negatives.to(device)
    
    # BEV point clouds have 1 channel, but ResNet expects 3 channels
    # Expand from [batch_size, 1, H, W] to [batch_size, 3, H, W]
    anchors_expanded = anchors.expand(-1, 3, -1, -1)
    positives_expanded = positives.expand(-1, 3, -1, -1)
    negatives_expanded = negatives.expand(-1, 3, -1, -1)

        
    # Get embeddings from the base model
    anchor_embeddings = base_model(anchors_expanded)
    positive_embeddings = base_model(positives_expanded)
    negative_embeddings = base_model(negatives_expanded)

    return anchor_embeddings, positive_embeddings, negative_embeddings

def get_embeddings_dino(anchors, positives, negatives, padding, base_model, device):
       
    # Apply padding to make dimensions compatible with patch size
    anchors = torch.stack([F.pad(x, padding, value=0) for x in anchors])
    positives = torch.stack([F.pad(x, padding, value=0) for x in positives])
    negatives = torch.stack([F.pad(x, padding, value=0) for x in negatives])
    
    # Move to device
    anchors = anchors.to(device)
    positives = positives.to(device)
    negatives = negatives.to(device)
    
    # BEV point clouds have 1 channel, but DINOv2 expects 3 channels
    # Expand from [batch_size, 1, H, W] to [batch_size, 3, H, W]
    anchors_expanded = anchors.expand(-1, 3, -1, -1)
    positives_expanded = positives.expand(-1, 3, -1, -1)
    negatives_expanded = negatives.expand(-1, 3, -1, -1)
    
    # Get embeddings from the base model

    anchor_embeddings = base_model(anchors_expanded).last_hidden_state[:, 1:, :].permute(0,2,1).reshape(-1,768,36,36)
    positive_embeddings = base_model(positives_expanded).last_hidden_state[:, 1:, :].permute(0,2,1).reshape(-1,768,36,36)
    negative_embeddings = base_model(negatives_expanded).last_hidden_state[:, 1:, :].permute(0,2,1).reshape(-1,768,36,36)

    return anchor_embeddings, positive_embeddings, negative_embeddings



def get_embeddings_dino_single(imgs, padding, base_model, device):
       
    # Apply padding to make dimensions compatible with patch size
    imgs = torch.stack([F.pad(x, padding, value=0) for x in imgs])


    
    # BEV point clouds have 1 channel, but DINOv2 expects 3 channels
    # Expand from [batch_size, 1, H, W] to [batch_size, 3, H, W]
    imgs_expanded = imgs.expand(-1, 3, -1, -1)

    
    # Get embeddings from the base model

    imgs_embeddings = base_model(imgs_expanded).last_hidden_state[:, 1:, :].permute(0,2,1).reshape(-1,768,36,36)


    return imgs