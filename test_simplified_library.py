"""
Test the simplified forestry_carbon_arr library focused on import/path management.
"""

import sys
from pathlib import Path

# Add the library to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_simplified_library():
    """Test the simplified library functionality."""
    print("🌳 Testing Simplified forestry_carbon_arr Library")
    print("=" * 60)
    
    try:
        # Test 1: Import main class
        print("\n1. Testing imports...")
        try:
            from forestry_carbon_arr import ForestryCarbonARR, ConfigManager
            print("   ✅ Successfully imported ForestryCarbonARR and ConfigManager")
        except ImportError:
            # Try direct import
            from core.main import ForestryCarbonARR
            from config.config_manager import ConfigManager
            print("   ✅ Successfully imported ForestryCarbonARR and ConfigManager (direct import)")
        
        # Test 2: Initialize without auto-setup
        print("\n2. Testing initialization...")
        forestry = ForestryCarbonARR(auto_setup=False)
        print("   ✅ ForestryCarbonARR initialized successfully")
        
        # Test 3: Manual setup
        print("\n3. Testing manual setup...")
        forestry.setup()
        print("   ✅ Setup completed")
        
        # Test 4: Check system info
        print("\n4. Testing system info...")
        system_info = forestry.get_system_info()
        print(f"   Version: {system_info.get('version', 'unknown')}")
        print(f"   GEE Forestry available: {system_info.get('gee_forestry_available', False)}")
        print(f"   Import strategy: {system_info.get('import_strategy', 'none')}")
        
        # Test 5: Check properties
        print("\n5. Testing properties...")
        print(f"   GEE Forestry available: {forestry.gee_forestry_available}")
        print(f"   GEE Forestry path: {forestry.gee_forestry_path}")
        print(f"   Import strategy: {forestry.import_strategy}")
        
        # Test 6: Import guide
        print("\n6. Testing import guide...")
        guide = forestry.get_import_guide()
        if "GEE_notebook_Forestry" in guide:
            print("   ✅ Import guide generated successfully")
        else:
            print("   ⚠️  Import guide indicates GEE_notebook_Forestry not available")
        
        # Test 7: Dependency manager
        print("\n7. Testing dependency manager...")
        dep_status = forestry.dependency_manager.get_dependency_status()
        print(f"   Core dependencies: {dep_status['core']['available']}")
        print(f"   GEE dependencies: {dep_status['gee']['available']}")
        print(f"   GEE Forestry modules: {dep_status['gee_forestry']['available']}")
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_import_manager():
    """Test the import manager directly."""
    print(f"\n🔧 Testing Import Manager Directly")
    print("=" * 40)
    
    try:
        from utils.import_manager import ImportManager
        
        import_manager = ImportManager()
        success, path, strategy = import_manager.detect_and_setup_gee_forestry()
        
        if success:
            print(f"   ✅ GEE_notebook_Forestry detected: {path}")
            print(f"   📋 Strategy: {strategy}")
            
            # Get import info
            import_info = import_manager.get_import_info()
            available_modules = import_info['available_modules']
            
            successful_imports = sum(available_modules.values())
            total_modules = len(available_modules)
            
            print(f"   📊 Modules available: {successful_imports}/{total_modules}")
            
            return True
        else:
            print(f"   ⚠️  GEE_notebook_Forestry not detected")
            return True  # This is acceptable
            
    except Exception as e:
        print(f"   ❌ Import manager test failed: {e}")
        return False

def show_library_structure():
    """Show the simplified library structure."""
    print(f"\n📁 Simplified Library Structure")
    print("=" * 40)
    
    structure = """
forestry_carbon_arr/
├── __init__.py                 # Main exports
├── core/
│   ├── __init__.py
│   └── main.py                 # ForestryCarbonARR class
├── config/
│   ├── __init__.py
│   ├── config_manager.py       # Configuration management
│   └── default_config.py       # Default configuration
├── utils/
│   ├── __init__.py
│   ├── dependency_manager.py   # Dependency checking
│   ├── path_resolver.py        # Path resolution
│   └── import_manager.py       # Import management
├── exceptions.py               # Custom exceptions
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
└── README.md                   # Documentation
"""
    
    print(structure)

if __name__ == "__main__":
    # Show library structure
    show_library_structure()
    
    # Test simplified library
    library_test = test_simplified_library()
    
    # Test import manager
    import_test = test_import_manager()
    
    # Summary
    print(f"\n📊 Test Results:")
    print(f"   Library test: {'✅' if library_test else '❌'}")
    print(f"   Import manager test: {'✅' if import_test else '❌'}")
    
    if library_test and import_test:
        print(f"\n🎉 SIMPLIFIED LIBRARY TEST: PASSED")
        print(f"✅ forestry_carbon_arr is working correctly")
        print(f"✅ Focused on import/path management only")
        print(f"✅ Ready for workflow implementation")
    else:
        print(f"\n❌ SIMPLIFIED LIBRARY TEST: FAILED")
        print(f"❌ Check the errors above")
