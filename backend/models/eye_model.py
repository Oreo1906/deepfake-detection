"""
Eye Model - Detects eye artifacts in deepfake images
Simple EfficientNet_B0 classifier (no multi-head)
"""

from torchvision.models import efficientnet_b0
import torch.nn as nn


class EyeModel(nn.Module):
    """
    Simple EfficientNet_B0 classifier
    
    Input: RGB image tensor (B, 3, 224, 224)
    Output: (B, 2) - Real/Fake classification logits
    """
    
    def __init__(self, dropout=0.4):
        super().__init__()
        backbone = efficientnet_b0(weights=None)
        in_feat  = backbone.classifier[1].in_features
        
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_feat, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.avgpool(self.features(x))
        x = x.flatten(1)
        x = self.classifier(x)
        return x  # Just logits, no dict for this simpler model

