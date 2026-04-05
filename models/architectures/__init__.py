"""
DeepFake Detection Models - Architecture Definitions
PyTorch models for deepfake detection with multiple specialized heads
"""

from torchvision.models import efficientnet_b0
import torch.nn as nn

__all__ = ['EyeModel', 'LipModel', 'NoseModel', 'SkinModel', 'GeometryModel']
