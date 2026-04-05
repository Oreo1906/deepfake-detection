"""
Nose Model - Detects nose region artifacts in deepfake images
Four-headed architecture: classification + geometry + texture + artifact
"""

from torchvision.models import efficientnet_b0
import torch.nn as nn


class NoseModel(nn.Module):
    """
    4 heads: classifier + geometry + texture + artifact
    
    Input: RGB image tensor (B, 3, 224, 224)
    Output: Dict with keys:
        - "logits": (B, 2) - Real/Fake classification logits
        - "geometry": (B,) - Nose geometry anomaly score [0-1]
        - "texture": (B,) - Nose texture anomaly score [0-1]
        - "artifact": (B,) - Nose artifact score [0-1]
    """
    
    def __init__(self, dropout=0.4):
        super().__init__()
        bb = efficientnet_b0(weights=None)
        in_feat = bb.classifier[1].in_features
        self.features = bb.features
        self.avgpool  = bb.avgpool
        self.shared   = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_feat, 512), 
            nn.BatchNorm1d(512), 
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout/2), 
            nn.Linear(512, 128),
            nn.BatchNorm1d(128), 
            nn.ReLU(True),
            nn.Dropout(p=dropout/4), 
            nn.Linear(128, 2)
        )
        self.geometry_head = nn.Sequential(
            nn.Dropout(p=dropout/2), 
            nn.Linear(512, 64),
            nn.ReLU(True), 
            nn.Linear(64, 1), 
            nn.Sigmoid()
        )
        self.texture_head = nn.Sequential(
            nn.Dropout(p=dropout/2), 
            nn.Linear(512, 64),
            nn.ReLU(True), 
            nn.Linear(64, 1), 
            nn.Sigmoid()
        )
        self.artifact_head = nn.Sequential(
            nn.Dropout(p=dropout/2), 
            nn.Linear(512, 64),
            nn.ReLU(True), 
            nn.Linear(64, 1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        f = self.avgpool(self.features(x)).flatten(1)
        s = self.shared(f)
        return {
            "logits":   self.classifier(s),
            "geometry": self.geometry_head(s).squeeze(1),
            "texture":  self.texture_head(s).squeeze(1),
            "artifact": self.artifact_head(s).squeeze(1)
        }
