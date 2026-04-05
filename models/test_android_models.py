"""
Test converted TorchScript models
Loads .pt files and runs inference to verify they work correctly
"""

import torch
from pathlib import Path


def test_model(model_path, example_input, model_name):
    """Test a TorchScript model"""
    print(f"\nTesting: {model_name}")
    print(f"{'='*50}")
    
    try:
        # Load TorchScript model
        model = torch.jit.load(str(model_path))
        model.eval()
        print(f"✓ Loaded model: {model_path.name}")
        
        # Run inference
        with torch.no_grad():
            if isinstance(example_input, tuple):
                output = model(*example_input)
            else:
                output = model(example_input)
        
        # Print output structure
        if isinstance(output, dict):
            print(f"✓ Output keys: {list(output.keys())}")
            for key, val in output.items():
                if isinstance(val, torch.Tensor):
                    print(f"  - {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"✓ Output shape: {output.shape}")
        
        print(f"✓ Model works correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    android_models_dir = Path(__file__).parent.parent / 'android_models'
    
    print(f"\n{'#'*50}")
    print(f"# Testing TorchScript Models")
    print(f"#{'#'*50}\n")
    print(f"Models directory: {android_models_dir}\n")
    
    # Example inputs
    example_img = torch.randn(1, 3, 224, 224)
    example_geometry = torch.randn(1, 52)
    
    success_count = 0
    total_count = 0
    
    # Test each model
    models_to_test = [
        ('eye_model.pt', example_img, 'Eye Model'),
        ('lip_model.pt', example_img, 'Lip Model'),
        ('nose_model.pt', example_img, 'Nose Model'),
        ('skin_model.pt', (example_img, example_img, example_img), 'Skin Model'),
        ('geometry_model.pt', example_geometry, 'Geometry Model'),
    ]
    
    for model_file, input_data, name in models_to_test:
        model_path = android_models_dir / model_file
        if model_path.exists():
            total_count += 1
            if test_model(model_path, input_data, name):
                success_count += 1
        else:
            print(f"\n✗ Model not found: {model_file}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Passed: {success_count}/{total_count} models")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()
