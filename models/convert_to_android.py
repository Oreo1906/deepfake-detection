"""
Convert all models to Android-compatible formats.
Two formats:
  A) TorchScript (.ptl)  → PyTorch Mobile for Android
  B) ONNX (.onnx) → For potential TFLite conversion
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from pathlib import Path
import sys

# Add architectures to path
sys.path.append(str(Path(__file__).parent / 'architectures'))

from eye_model import EyeModel
from lip_model import LipModel
from nose_model import NoseModel
from skin_model import SkinModel
from geometry_model import GeometryClassifier

DEVICE = torch.device('cpu')  # Always export on CPU for mobile

# ─────────────────────────────────────────────
# SINGLE-STREAM WRAPPER
# Needed because Android can't handle dict outputs.
# Wraps model to output a single tensor: [real_prob, fake_prob, score1, score2...]
# ─────────────────────────────────────────────
class SimpleClassifierWrapper(nn.Module):
    """
    Wraps simple classifiers that output logits directly (like EyeModel).
    Input : (1, 3, 224, 224)
    Output: (1, 2) → [real_prob, fake_prob]
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)  # (B, 2)
        return probs


class SingleStreamWrapper(nn.Module):
    """
    Wraps Eye / Lip / Nose models.
    Input : (1, 3, 224, 224)
    Output: (1, 4+) → [real_prob, fake_prob, aux_score_1, aux_score_2, ...]
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out   = self.model(x)
        probs = torch.softmax(out["logits"], dim=1)  # (B, 2)
        # Collect aux scores — handle variable number of heads
        aux = []
        for key in ["geometry","texture","artifact","frequency","symmetry","deviation"]:
            if key in out:
                aux.append(out[key].unsqueeze(1))
        if aux:
            return torch.cat([probs] + aux, dim=1)
        return probs


class TripleStreamWrapper(nn.Module):
    """
    Wraps SkinModel (3 inputs).
    Input : rgb(1,3,224,224), hf(1,3,224,224), lap(1,3,224,224)
    Output: (1, 5) → [real_prob, fake_prob, texture, artifact, frequency]
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, rgb, hf, lap):
        out   = self.model(rgb, hf, lap)
        probs = torch.softmax(out["logits"], dim=1)
        return torch.cat([
            probs,
            out["texture"].unsqueeze(1),
            out["artifact"].unsqueeze(1),
            out["frequency"].unsqueeze(1),
        ], dim=1)


class GeometryWrapper(nn.Module):
    """
    Wraps GeometryClassifier (52-dim vector input).
    Input : (1, 52)
    Output: (1, 4) → [real_prob, fake_prob, symmetry, deviation]
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out   = self.model(x)
        probs = torch.softmax(out["logits"], dim=1)
        return torch.cat([
            probs,
            out["symmetry"].unsqueeze(1),
            out["deviation"].unsqueeze(1),
        ], dim=1)


# ── FORMAT A: TorchScript (.ptl) — PyTorch Mobile ─────────────────
def export_torchscript(model, dummy_input, out_path):
    """
    .ptl files work directly with PyTorch Mobile on Android.
    No extra dependencies needed — just add pytorch_android to build.gradle.
    """
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input)
    
    # Optimize for mobile
    try:
        optimized = torch.utils.mobile_optimizer.optimize_for_mobile(traced)
        optimized._save_for_lite_interpreter(str(out_path))
    except AttributeError:
        # Fallback if mobile optimizer not available
        traced.save(str(out_path))
    
    size_mb = Path(out_path).stat().st_size / 1024 / 1024
    print(f"  ✅ TorchScript saved: {out_path.name}  ({size_mb:.1f} MB)")


# ── FORMAT B: ONNX ───────────────────────────────────────
def export_onnx(model, dummy_input, out_path, input_names, output_names):
    """Export to ONNX — intermediate step before TFLite conversion."""
    model.eval()
    torch.onnx.export(
        model, dummy_input, str(out_path),
        export_params=True, opset_version=12,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={name: {0: 'batch'} for name in input_names + output_names}
    )
    size_mb = Path(out_path).stat().st_size / 1024 / 1024
    print(f"  ✅ ONNX saved: {out_path.name}  ({size_mb:.1f} MB)")


# ─────────────────────────────────────────────
# EXPORT ALL MODELS
# ─────────────────────────────────────────────
def export_all():
    base_dir = Path(__file__).parent.parent
    out_dir = base_dir / 'android_models'
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# PyTorch to Android Conversion (Mobile-Optimized)")
    print(f"#{'#'*60}\n")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {out_dir}\n")

    configs = [
        {
            "name":       "eye",
            "model_cls":  EyeModel,
            "pth":        base_dir / "eye_model.pth",
            "wrapper":    SimpleClassifierWrapper,  # Changed wrapper
            "dummy":      torch.randn(1, 3, 224, 224),
            "in_names":   ["input"],
            "out_names":  ["output"],
        },
        {
            "name":       "lip",
            "model_cls":  LipModel,
            "pth":        base_dir / "lip_model.pth",
            "wrapper":    SingleStreamWrapper,
            "dummy":      torch.randn(1, 3, 224, 224),
            "in_names":   ["input"],
            "out_names":  ["output"],
        },
        {
            "name":       "nose",
            "model_cls":  NoseModel,
            "pth":        base_dir / "nose_model.pth",
            "wrapper":    SingleStreamWrapper,
            "dummy":      torch.randn(1, 3, 224, 224),
            "in_names":   ["input"],
            "out_names":  ["output"],
        },
        {
            "name":       "skin",
            "model_cls":  SkinModel,
            "pth":        base_dir / "skin_model.pth",
            "wrapper":    TripleStreamWrapper,
            "dummy":      (torch.randn(1,3,224,224),
                          torch.randn(1,3,224,224),
                          torch.randn(1,3,224,224)),
            "in_names":   ["rgb","hf","lap"],
            "out_names":  ["output"],
        },
        {
            "name":       "geometry",
            "model_cls":  GeometryClassifier,
            "pth":        base_dir / "geometry_model.pth",
            "wrapper":    GeometryWrapper,
            "dummy":      torch.randn(1, 52),
            "in_names":   ["features"],
            "out_names":  ["output"],
        },
    ]

    success_count = 0
    total_count = 0

    for cfg in configs:
        name = cfg["name"]
        pth  = cfg["pth"]

        print(f"\n{'='*60}")
        print(f"Converting: {name.upper()} Model")
        print(f"{'='*60}")

        if not pth.exists():
            print(f"⚠️  Skipping — checkpoint not found: {pth.name}")
            continue

        total_count += 1

        try:
            # Load weights
            base_model = cfg["model_cls"]()
            checkpoint = torch.load(str(pth), map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    base_model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    base_model.load_state_dict(checkpoint['state_dict'])
                else:
                    base_model.load_state_dict(checkpoint)
            else:
                base_model.load_state_dict(checkpoint)
            
            base_model.eval()
            print(f"✓ Loaded checkpoint: {pth.name}")

            # Wrap for mobile output format
            wrapped = cfg["wrapper"](base_model)
            wrapped.eval()

            dummy = cfg["dummy"]

            # Test wrapped output
            with torch.no_grad():
                out = wrapped(*dummy) if isinstance(dummy, tuple) else wrapped(dummy)
            print(f"✓ Tested wrapper - Output shape: {out.shape}")
            print(f"  Sample: real_prob={out[0,0]:.4f}, fake_prob={out[0,1]:.4f}")

            # Export TorchScript (.ptl)
            ptl_path = out_dir / f"{name}_model.ptl"
            try:
                export_torchscript(wrapped, dummy, ptl_path)
            except Exception as e:
                print(f"  ⚠️  TorchScript export failed: {e}")

            # Export ONNX
            onnx_path = out_dir / f"{name}_model.onnx"
            try:
                export_onnx(
                    wrapped, dummy, onnx_path,
                    cfg["in_names"], cfg["out_names"]
                )
            except Exception as e:
                print(f"  ⚠️  ONNX export failed: {e}")

            success_count += 1

        except Exception as e:
            print(f"✗ Conversion failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print(f"CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully converted: {success_count}/{total_count} models")
    print(f"Output directory: {out_dir}")
    
    if success_count > 0:
        print(f"\n📱 Android output tensor format:")
        print(f"  index 0 → real_prob  (0.0-1.0)")
        print(f"  index 1 → fake_prob  (0.0-1.0)")
        print(f"  index 2+ → aux scores (geometry/texture/artifact etc.)")
        print(f"\n✓ Models ready for Android integration!")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    export_all()
