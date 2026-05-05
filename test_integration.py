"""
Simple test to verify GEE_notebook_Forestry/osi integration.
"""

import sys
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gee_forestry_access():
    """Test if GEE_notebook_Forestry/osi is accessible."""
    print("🔍 Testing GEE_notebook_Forestry/osi accessibility...")
    print("=" * 60)
    
    # Test 1: Check if GEE_notebook_Forestry directory exists
    print("\n1. Checking for GEE_notebook_Forestry directory...")
    
    possible_paths = [
        # Local development paths
        Path.cwd() / "GEE_notebook_Forestry",
        Path.cwd().parent / "GEE_notebook_Forestry",
        Path(__file__).parent.parent / "GEE_notebook_Forestry",
        
        # Container paths
        Path("/usr/src/app/gee_lib"),
        Path("/app/gee_lib"),
        Path("/usr/src/app/GEE_notebook_Forestry"),
        Path("/app/GEE_notebook_Forestry"),
        
        # Environment variable path
        Path(os.environ.get('GEE_FORESTRY_PATH', '')) if os.environ.get('GEE_FORESTRY_PATH') else None,
    ]
    
    gee_forestry_path = None
    for path in possible_paths:
        if path and path.exists():
            print(f"   ✅ Found GEE_notebook_Forestry at: {path}")
            gee_forestry_path = path
            break
    
    if not gee_forestry_path:
        print("   ❌ GEE_notebook_Forestry directory not found")
        print("   Searched paths:")
        for path in possible_paths:
            if path:
                print(f"     - {path}")
        return False
    
    # Test 2: Check if osi directory exists
    print(f"\n2. Checking for osi directory in {gee_forestry_path}...")
    osi_path = gee_forestry_path / "osi"
    if osi_path.exists():
        print(f"   ✅ Found osi directory at: {osi_path}")
    else:
        print(f"   ❌ osi directory not found in {gee_forestry_path}")
        return False
    
    # Test 3: Check if osi modules exist
    print(f"\n3. Checking for osi modules...")
    required_modules = [
        "osi/__init__.py",
        "osi/image_collection/main.py",
        "osi/spectral_indices/spectral_analysis.py",
        "osi/fcd/main_fcd.py",
        "osi/ml/main.py",
        "osi/utils/main.py"
    ]
    
    missing_modules = []
    for module in required_modules:
        module_path = gee_forestry_path / module
        if module_path.exists():
            print(f"   ✅ {module}")
        else:
            print(f"   ❌ {module}")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n   Missing modules: {missing_modules}")
        return False
    
    # Test 4: Try to add to Python path and import
    print(f"\n4. Testing Python import...")
    try:
        # Add to Python path
        sys.path.insert(0, str(gee_forestry_path))
        print(f"   ✅ Added {gee_forestry_path} to Python path")
        
        # Try to import osi
        import osi
        print(f"   ✅ Successfully imported osi module")
        
        # Try to import specific modules
        test_imports = [
            "osi.image_collection.main",
            "osi.spectral_indices.spectral_analysis", 
            "osi.fcd.main_fcd",
            "osi.ml.main",
            "osi.utils.main"
        ]
        
        for module_name in test_imports:
            try:
                module = __import__(module_name)
                print(f"   ✅ Successfully imported {module_name}")
            except ImportError as e:
                print(f"   ❌ Failed to import {module_name}: {e}")
                return False
        
        print(f"\n🎉 All tests passed! GEE_notebook_Forestry/osi is accessible.")
        return True
        
    except Exception as e:
        print(f"   ❌ Import test failed: {e}")
        return False

def test_environment_info():
    """Display environment information."""
    print(f"\n📋 Environment Information:")
    print(f"   Python version: {sys.version}")
    print(f"   Current working directory: {Path.cwd()}")
    print(f"   Script location: {Path(__file__).parent}")
    print(f"   Python path: {sys.path[:3]}...")  # Show first 3 entries
    
    # Check environment variables
    env_vars = ['GEE_FORESTRY_PATH', 'PYTHONPATH', 'CONTAINER']
    print(f"   Environment variables:")
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"     {var}: {value}")

if __name__ == "__main__":
    print("🌳 GEE_notebook_Forestry Integration Test")
    print("=" * 60)
    
    # Display environment info
    test_environment_info()
    
    # Test accessibility
    success = test_gee_forestry_access()
    
    if success:
        print(f"\n✅ Integration test PASSED")
        print(f"\nNext steps:")
        print(f"   1. The forestry_carbon_arr library can now integrate with GEE_notebook_Forestry")
        print(f"   2. You can proceed with implementing the main workflow")
    else:
        print(f"\n❌ Integration test FAILED")
        print(f"\nTroubleshooting:")
        print(f"   1. Make sure GEE_notebook_Forestry is in the correct location")
        print(f"   2. Check that all required osi modules exist")
        print(f"   3. Verify Python path configuration")
        print(f"   4. For container usage, ensure proper volume mounting")
