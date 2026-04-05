"""
Convert PyTorch models to TorchScript for Android deployment
Loads trained .pth checkpoints and exports to .pt format
"""

import torch
import sys
from pathlib import Path

# Add architectures to path
sys.path.append(str(Path(__file__).parent / 'architectures'))

from eye_model import EyeModel
from lip_model import LipModel
from nose_model import NoseModel
from skin_model import SkinModel
from geometry_model import GeometryClassifier


def convert_model(model, checkpoint_path, output_path, example_input):
    """
    Convert a PyTorch model to TorchScript
    
    Args:
        model: PyTorch model instance
        checkpoint_path: Path to .pth checkpoint file
        output_path: Path to save .pt TorchScript file
        example_input: Example input tensor(s) for tracing
    """
    print(f"\n{'='*60}")
    print(f"Converting: {checkpoint_path.name}")
    print(f"{'='*60}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded state_dict from checkpoint")
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"✓ Loaded state_dict from checkpoint")
        else:
            model.load_state_dict(checkpoint)
            print(f"✓ Loaded checkpoint directly")
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        return False
    
    # Set to evaluation mode
    model.eval()
    
    # Convert to TorchScript
    try:
        with torch.no_grad():
            if isinstance(example_input, tuple):
                traced_model = torch.jit.trace(model, example_input)
            else:
                traced_model = torch.jit.trace(model, (example_input,))
        print(f"✓ Traced model successfully")
    except Exception as e:
        print(f"✗ Error tracing model: {e}")
        return False
    
    # Save TorchScript model
    try:
        traced_model.save(str(output_path))
        file_size_mb = output_path.stat().st_size / (1024*1024)
        print(f"✓ Saved to: {output_path.name} ({file_size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        return False


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    checkpoints_dir = base_dir
    output_dir = base_dir / 'android_models'
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# PyTorch to TorchScript Conversion")
    print(f"#{'#'*60}\n")
    print(f"Checkpoints directory: {checkpoints_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Example inputs (batch_size=1, channels=3, height=224, width=224)
    example_img = torch.randn(1, 3, 224, 224)
    example_geometry = torch.randn(1, 52)
    
    success_count = 0
    total_count = 0
    
    # 1. Eye Model
    total_count += 1
    if (checkpoints_dir / 'eye_model.pth').exists():
        eye_model = EyeModel(dropout=0.4)
        if convert_model(
            eye_model,
            checkpoints_dir / 'eye_model.pth',
            output_dir / 'eye_model.pt',
            example_img
        ):
            success_count += 1
    else:
        print(f"✗ Checkpoint not found: eye_model.pth")
    
    # 2. Lip Model
    total_count += 1
    if (checkpoints_dir / 'lip_model.pth').exists():
        lip_model = LipModel(dropout=0.4)
        if convert_model(
            lip_model,
            checkpoints_dir / 'lip_model.pth',
            output_dir / 'lip_model.pt',
            example_img
        ):
            success_count += 1
    else:
        print(f"✗ Checkpoint not found: lip_model.pth")
    
    # 3. Nose Model
    total_count += 1
    if (checkpoints_dir / 'nose_model.pth').exists():
        nose_model = NoseModel(dropout=0.4)
        if convert_model(
            nose_model,
            checkpoints_dir / 'nose_model.pth',
            output_dir / 'nose_model.pt',
            example_img
        ):
            success_count += 1
    else:
        print(f"✗ Checkpoint not found: nose_model.pth")
    
    # 4. Skin Model (requires 3 inputs)
    total_count += 1
    if (checkpoints_dir / 'skin_model.pth').exists():
        skin_model = SkinModel(dropout=0.4)
        if convert_model(
            skin_model,
            checkpoints_dir / 'skin_model.pth',
            output_dir / 'skin_model.pt',
            (example_img, example_img, example_img)  # rgb, hf, lap
        ):
            success_count += 1
    else:
        print(f"✗ Checkpoint not found: skin_model.pth")
    
    # 5. Geometry Model
    total_count += 1
    if (checkpoints_dir / 'geometry_model.pth').exists():
        geometry_model = GeometryClassifier(input_dim=52, dropout=0.3)
        if convert_model(
            geometry_model,
            checkpoints_dir / 'geometry_model.pth',
            output_dir / 'geometry_model.pt',
            example_geometry
        ):
            success_count += 1
    else:
        print(f"✗ Checkpoint not found: geometry_model.pth")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully converted: {success_count}/{total_count} models")
    print(f"Output directory: {output_dir}")
    print(f"\nConverted models are ready for Android deployment!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
