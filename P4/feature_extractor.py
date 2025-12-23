import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision import models, transforms

class MobileNetV2Extractor(BaseFeaturesExtractor):
    """
    Feature extractor using a pre-trained MobileNetV2.
    It takes an image reference (resized to 224x224 usually) and returns a feature vector.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 1280):
        super().__init__(observation_space, features_dim)
        
        # Load pre-trained MobileNetV2
        # weights='DEFAULT' loads the best available weights
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Remove the classifier (the last layer)
        # MobileNetV2 structure: features -> avgpool -> classifier
        # We want the output of features + avgpool
        self.features = self.mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Freeze weights - We only want to use it as a fixed feature extractor (SRL)
        for param in self.mobilenet.parameters():
            param.requires_grad = False
            
        # Optional: Unfreeze last layers if fine-tuning is needed later
        # for param in self.features[-2:].parameters():
        #     param.requires_grad = True

        # Preprocessing transforms (normalization expected by ImageNet models)
        self.transforms = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        observations: Tensor of shape (batch, C, H, W) in range [0, 255] or [0, 1] depends on env.
        SB3 sends observations in [0, 1] if generated from gym.spaces.Box(low=0, high=255) wrapped.
        Assume standard SB3 preprocessing (divides by 255 if configured).
        """
        
        # Ensure input is what torch expects (Batch, Channel, Height, Width)
        # And values are normalized for the pre-trained model
        # NOTE: SB3 CnnPolicy might already pass normalized [0,1] floats. 
        # ImageNet models expect Normalized with mean/std.
        
        x = observations
        
        # Apply normalization (assuming x is already [0, 1])
        # If x is [0, 255], we should divide. 
        # Usually SB3 handles scaling if we tell it.
        
        x = self.transforms(x)
        
        # Extract features
        x = self.features(x)         # -> (Batch, 1280, 7, 7) or similar
        x = self.avgpool(x)          # -> (Batch, 1280, 1, 1)
        x = torch.flatten(x, 1)      # -> (Batch, 1280)
        
        return x
