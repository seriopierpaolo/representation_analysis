import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights


class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50Encoder, self).__init__()
        # Load pretrained ResNet50 without the final FC layer
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Use only the feature extraction part of ResNet50
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # Output: 2048×16×16
        
        return x


class LightBEVDecoder(nn.Module):
    def __init__(self, output_size=(256, 256)):
        super(LightBEVDecoder, self).__init__()
        
        self.output_size = output_size
        
        # Decoder blocks - progressive upsampling
        # Block 1: 2048×16×16 -> 512×32×32
        self.decoder_block1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Block 2: 512×32×32 -> 256×64×64
        self.decoder_block2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Block 3: 256×64×64 -> 128×128×128
        self.decoder_block3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Block 4: 128×128×128 -> 64×256×256
        self.decoder_block4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        
        # Final layer: 64×256×256 -> 1×256×256 (grayscale image)
        self.final_layer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            #nn.Sigmoid(),  # Sigmoid to ensure output values are in [0,1] for grayscale

        )
        
    def forward(self, x):
        x = self.decoder_block1(x)
        x = self.decoder_block2(x)
        x = self.decoder_block3(x)
        x = self.decoder_block4(x)
        #x = self.decoder_block5(x)
        x = self.final_layer(x)
        output = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        
        return output


class ResNet50BEVModel(nn.Module):
    def __init__(self, pretrained_encoder=True, output_size=(256, 256)):
        super(ResNet50BEVModel, self).__init__()
        self.encoder = ResNet50Encoder(pretrained=pretrained_encoder)
        self.decoder = LightBEVDecoder(output_size=output_size)
        
    def forward(self, x):
        features = self.encoder(x)
        bev_output = self.decoder(features)

        return bev_output, features