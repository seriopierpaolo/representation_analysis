import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, input_size=512, target_size=512):
        # Projection Head for Contrastive Loss
        super(ProjectionHead, self).__init__()
        self.input_size = input_size
        self.target_size = target_size
        self.projection = nn.Sequential(
            
            nn.Conv2d(1024, 512, kernel_size=2, stride=3, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            #nn.BatchNorm2d(512),  # Use number of channels (1) instead of input_size
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(512, 128, kernel_size=2, stride=3, padding=0),
            #nn.BatchNorm2d(128),  # Use number of channels (1) instead of input_size
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=4, stride=3, padding=0),
            #nn.BatchNorm1d(128),  # Use number of channels (1) instead of input_size
            nn.LeakyReLU(inplace=True),
            
        )
        
        self.bn = torch.nn.BatchNorm1d(128)

    def forward(self, x):
        # Reshape input to [batch_size, channels, sequence_length]
        #x = x.squeeze(0)  # Add channel dimension
        x = self.projection(x)
        x = x.view(128)  # Reshape to (N, 128, L)
        
        #x = self.bn(x)
        #x.squeeze()
        return x

def test_projectionhead(pretrained_weights=None, input_size=500, target_size=500):
    # Create model
    model = ProjectionHead(input_size=input_size, target_size=target_size)
    # Compile (in PyTorch, we typically set up optimizer during training)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()
    # Load pretrained weights if provided
    if pretrained_weights:
        model.load_state_dict(torch.load(pretrained_weights))
    return model, optimizer, criterion

# Example usage
if __name__ == "__main__":
    # Create model for input_size 2048
    model, optimizer, criterion = test_projectionhead(input_size=1042)
    # Example input tensor
    x = torch.randn(1, 1024, 32, 32)
    # Forward pass
    output = model(x)
    #output.unsqueeze(0)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")