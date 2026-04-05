"""
Geometry Model - Analyzes facial geometry features for deepfake detection
MLP-based architecture with 3 heads: classification + symmetry + deviation
"""

import torch.nn as nn


class GeometryClassifier(nn.Module):
    """
    MLP on 52-dim feature vector — 3 heads: classifier + symmetry + deviation
    
    Input: Feature vector (B, 52) - facial landmarks/geometry features
    Output: Dict with keys:
        - "logits": (B, 2) - Real/Fake classification logits
        - "symmetry": (B,) - Facial symmetry score [0-1]
        - "deviation": (B,) - Geometry deviation score [0-1]
    """
    
    def __init__(self, input_dim=52, dropout=0.3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(True), 
            nn.Dropout(p=dropout),
            nn.Linear(256, 128), 
            nn.BatchNorm1d(128),
            nn.ReLU(True), 
            nn.Dropout(p=dropout/2),
            nn.Linear(128, 64), 
            nn.BatchNorm1d(64), 
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout/2), 
            nn.Linear(64, 32),
            nn.ReLU(True), 
            nn.Linear(32, 2)
        )
        self.symmetry_head = nn.Sequential(
            nn.Linear(64, 32), 
            nn.ReLU(True),
            nn.Linear(32, 1), 
            nn.Sigmoid()
        )
        self.deviation_head = nn.Sequential(
            nn.Linear(64, 32), 
            nn.ReLU(True),
            nn.Linear(32, 1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        s = self.shared(x)
        return {
            "logits":    self.classifier(s),
            "symmetry":  self.symmetry_head(s).squeeze(1),
            "deviation": self.deviation_head(s).squeeze(1)
        }
