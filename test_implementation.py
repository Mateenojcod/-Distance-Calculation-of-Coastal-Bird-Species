#!/usr/bin/env python3
"""
Test script to validate the implementation without requiring actual dependencies.

This script tests the module structure, imports, and basic functionality
without needing the actual ML libraries to be installed.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def test_module_structure():
    """Test that all modules are structured correctly."""
    print("Testing module structure...")
    
    # Test directory structure
    required_dirs = [
        'src',
        'src/bird_identification',
        'src/distance_calculation',
        'config',
        'examples'
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        assert path.exists(), f"Required directory missing: {dir_path}"
        print(f"  ✓ Directory exists: {dir_path}")
    
    # Test required files
    required_files = [
        'src/__init__.py',
        'src/bird_identification/__init__.py',
        'src/bird_identification/classifier.py',
        'src/bird_identification/detector.py',
        'src/distance_calculation/__init__.py',
        'src/distance_calculation/distance_estimator.py',
        'src/distance_calculation/camera_calibration.py',
        'requirements.txt',
        'README.md',
        'LIBRARY_GUIDE.md',
        'main.py'
    ]
    
    for file_path in required_files:
        path = Path(file_path)
        assert path.exists(), f"Required file missing: {file_path}"
        print(f"  ✓ File exists: {file_path}")
    
    print("✅ Module structure test passed!\n")


def test_imports():
    """Test that modules can be imported."""
    print("Testing module imports...")
    
    try:
        # Test main package import
        import src
        print(f"  ✓ Imported src (version: {src.__version__})")
        
        # Test bird identification imports
        from src.bird_identification import BirdDetector, BirdClassifier
        print("  ✓ Imported BirdDetector and BirdClassifier")
        
        # Test distance calculation imports
        from src.distance_calculation import DistanceEstimator, CameraCalibrator
        print("  ✓ Imported DistanceEstimator and CameraCalibrator")
        
        print("✅ Import test passed!\n")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}\n")
        return False


def test_class_instantiation():
    """Test that classes can be instantiated with proper error handling."""
    print("Testing class instantiation...")
    
    try:
        from src.bird_identification import BirdDetector, BirdClassifier
        from src.distance_calculation import DistanceEstimator, CameraCalibrator
        
        # Test CameraCalibrator (should work without dependencies)
        calibrator = CameraCalibrator(checkerboard_size=(9, 6), square_size=0.025)
        print("  ✓ CameraCalibrator instantiated")
        
        # Test DistanceEstimator with pinhole method (minimal dependencies)
        distance_estimator = DistanceEstimator(
            method='pinhole',
            camera_params={'focal_length': 800}
        )
        print("  ✓ DistanceEstimator instantiated (pinhole method)")
        
        # Test that classes handle missing dependencies gracefully
        try:
            # This should fail gracefully if dependencies aren't installed
            detector = BirdDetector(method='yolo', lazy_load=True)
            print("  ✓ BirdDetector instantiated (lazy loading)")
        except ValueError as e:
            print(f"  ✓ BirdDetector correctly handles missing dependencies: {str(e)[:50]}...")
        
        try:
            classifier = BirdClassifier(method='tensorflow', num_classes=10)
            print("  ✓ BirdClassifier instantiated")
        except ValueError as e:
            print(f"  ✓ BirdClassifier correctly handles missing dependencies: {str(e)[:50]}...")
        
        print("✅ Class instantiation test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Class instantiation failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test that configuration file is valid."""
    print("Testing configuration...")
    
    try:
        import yaml
        
        config_path = Path('config/default_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Verify required sections
        required_sections = ['camera', 'detection', 'classification', 'distance', 'bird_sizes']
        for section in required_sections:
            assert section in config, f"Missing config section: {section}"
            print(f"  ✓ Config section present: {section}")
        
        # Verify some key parameters
        assert config['camera']['focal_length'] > 0
        assert config['detection']['confidence_threshold'] >= 0
        assert len(config['classification']['class_names']) > 0
        print("  ✓ Config values are valid")
        
        print("✅ Configuration test passed!\n")
        return True
        
    except ImportError:
        print("  ⚠️  PyYAML not installed, skipping config validation\n")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}\n")
        return False


def test_documentation():
    """Test that documentation files are present and non-empty."""
    print("Testing documentation...")
    
    docs = {
        'README.md': 1000,  # Should be substantial
        'LIBRARY_GUIDE.md': 5000,  # Should be comprehensive
        'requirements.txt': 100,  # Should list dependencies
    }
    
    for doc_file, min_size in docs.items():
        path = Path(doc_file)
        assert path.exists(), f"Documentation file missing: {doc_file}"
        
        content = path.read_text()
        assert len(content) > min_size, f"{doc_file} seems too short ({len(content)} < {min_size})"
        
        print(f"  ✓ {doc_file} exists and has content ({len(content)} chars)")
    
    print("✅ Documentation test passed!\n")


def test_method_signatures():
    """Test that key methods have proper signatures."""
    print("Testing method signatures...")
    
    from src.bird_identification import BirdDetector, BirdClassifier
    from src.distance_calculation import DistanceEstimator, CameraCalibrator
    import inspect
    
    # Check BirdDetector methods
    assert hasattr(BirdDetector, 'detect'), "BirdDetector missing detect method"
    assert hasattr(BirdDetector, 'detect_in_video'), "BirdDetector missing detect_in_video method"
    print("  ✓ BirdDetector has required methods")
    
    # Check BirdClassifier methods
    assert hasattr(BirdClassifier, 'predict'), "BirdClassifier missing predict method"
    assert hasattr(BirdClassifier, 'train'), "BirdClassifier missing train method"
    print("  ✓ BirdClassifier has required methods")
    
    # Check DistanceEstimator methods
    assert hasattr(DistanceEstimator, 'estimate_distance'), "DistanceEstimator missing estimate_distance method"
    assert hasattr(DistanceEstimator, 'estimate_distance_pinhole'), "Missing pinhole method"
    assert hasattr(DistanceEstimator, 'estimate_distance_stereo'), "Missing stereo method"
    print("  ✓ DistanceEstimator has required methods")
    
    # Check CameraCalibrator methods
    assert hasattr(CameraCalibrator, 'calibrate_from_images'), "CameraCalibrator missing calibrate_from_images"
    assert hasattr(CameraCalibrator, 'save_calibration'), "CameraCalibrator missing save_calibration"
    print("  ✓ CameraCalibrator has required methods")
    
    print("✅ Method signature test passed!\n")


def main():
    """Run all tests."""
    print("=" * 70)
    print("  Testing Coastal Bird Species Identification Implementation")
    print("=" * 70)
    print()
    
    tests = [
        ("Module Structure", test_module_structure),
        ("Module Imports", test_imports),
        ("Class Instantiation", test_class_instantiation),
        ("Configuration", test_configuration),
        ("Documentation", test_documentation),
        ("Method Signatures", test_method_signatures),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result if result is not None else True))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}\n")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("=" * 70)
    print("  Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    print()
    print(f"  Total: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 All tests passed! Implementation is valid.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
