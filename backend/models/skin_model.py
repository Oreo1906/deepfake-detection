from torchvision.models import efficientnet_b0
import torch
import torch.nn as nn


class SkinModel(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        bb_a = efficientnet_b0(weights=None)
        bb_b = efficientnet_b0(weights=None)
        bb_c = efficientnet_b0(weights=None)
        feat_dim = bb_a.classifier[1].in_features

        self.feat_a = bb_a.features
        self.pool_a = bb_a.avgpool
        self.feat_b = bb_b.features
        self.pool_b = bb_b.avgpool
        self.feat_c = bb_c.features
        self.pool_c = bb_c.avgpool

        self.fusion = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(p=dropout / 4),
            nn.Linear(128, 2),
        )
        self.texture_head = nn.Sequential(
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.artifact_head = nn.Sequential(
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.frequency_head = nn.Sequential(
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, rgb, hf, lap):
        fa = self.pool_a(self.feat_a(rgb)).flatten(1)
        fb = self.pool_b(self.feat_b(hf)).flatten(1)
        fc = self.pool_c(self.feat_c(lap)).flatten(1)
        s = self.fusion(torch.cat([fa, fb, fc], dim=1))
        return {
            "logits": self.classifier(s),
            "texture": self.texture_head(s).squeeze(1),
            "artifact": self.artifact_head(s).squeeze(1),
            "frequency": self.frequency_head(s).squeeze(1),
        }
