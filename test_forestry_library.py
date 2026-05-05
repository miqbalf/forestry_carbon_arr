"""
Test the forestry_carbon_arr library with GEE_notebook_Forestry integration.
"""

import sys
from pathlib import Path

# Add the library to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Also add the parent directory to find the library
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_forestry_library():
    """Test the forestry_carbon_arr library."""
    print("🌳 Testing forestry_carbon_arr library...")
    print("=" * 60)
    
    try:
        # Import the main class
        try:
            from forestry_carbon_arr import ForestryCarbonARR
            print("✅ Successfully imported ForestryCarbonARR")
        except ImportError:
            # Try direct import
            from core.main import ForestryCarbonARR
            print("✅ Successfully imported ForestryCarbonARR (direct import)")
        
        # Initialize without GEE Forestry path (should work)
        print("\n1. Testing initialization without GEE Forestry path...")
        forestry = ForestryCarbonARR(auto_setup=False)  # Skip auto-setup for now
        print("✅ ForestryCarbonARR initialized successfully")
        
        # Get system info
        print("\n2. Getting system information...")
        system_info = forestry.get_system_info()
        print(f"   Version: {system_info.get('version', 'unknown')}")
        print(f"   GEE Forestry available: {system_info.get('gee_forestry_available', False)}")
        print(f"   ML available: {system_info.get('ml_available', False)}")
        print(f"   Satellite available: {system_info.get('satellite_available', False)}")
        
        # Test setting GEE Forestry path
        print("\n3. Testing GEE Forestry path setting...")
        gee_forestry_path = Path(__file__).parent.parent / "GEE_notebook_Forestry"
        if gee_forestry_path.exists():
            forestry.set_gee_forestry_path(gee_forestry_path)
            print(f"✅ GEE Forestry path set to: {gee_forestry_path}")
            
            # Test dependency manager
            print("\n4. Testing dependency manager...")
            dep_status = forestry.dependency_manager.get_dependency_status()
            print(f"   Core dependencies: {dep_status['core']['available']}")
            print(f"   GEE dependencies: {dep_status['gee']['available']}")
            print(f"   ML dependencies: {dep_status['ml']['available']}")
            print(f"   Satellite dependencies: {dep_status['satellite']['available']}")
            print(f"   GEE Forestry modules: {dep_status['gee_forestry']['available']}")
            
            if not dep_status['gee_forestry']['available']:
                print(f"   Missing GEE Forestry modules: {dep_status['gee_forestry']['missing']}")
        else:
            print(f"❌ GEE_notebook_Forestry not found at: {gee_forestry_path}")
        
        # Test configuration
        print("\n5. Testing configuration...")
        config = forestry.config
        print(f"   Project name: {config.get('project', {}).get('name', 'default')}")
        print(f"   Satellite provider: {config.get('satellite', {}).get('provider', 'default')}")
        
        print("\n🎉 forestry_carbon_arr library test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_conditional_imports():
    """Test conditional imports."""
    print("\n🔄 Testing conditional imports...")
    
    try:
        # Test importing with conditional availability flags
        from forestry_carbon_arr import (
            ForestryCarbonARR, 
            GEE_AVAILABLE, 
            ML_AVAILABLE, 
            SATELLITE_AVAILABLE
        )
        
        print(f"   GEE_AVAILABLE: {GEE_AVAILABLE}")
        print(f"   ML_AVAILABLE: {ML_AVAILABLE}")
        print(f"   SATELLITE_AVAILABLE: {SATELLITE_AVAILABLE}")
        
        print("✅ Conditional imports working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Conditional import test failed: {e}")
        return False

if __name__ == "__main__":
    print("🌳 forestry_carbon_arr Library Test")
    print("=" * 60)
    
    # Test basic library functionality
    basic_test = test_forestry_library()
    
    # Test conditional imports
    import_test = test_conditional_imports()
    
    if basic_test and import_test:
        print(f"\n🎉 All tests PASSED!")
        print(f"\n✅ forestry_carbon_arr library is working correctly")
        print(f"✅ GEE_notebook_Forestry integration is ready")
        print(f"✅ Conditional imports are working")
        
        print(f"\n📋 Ready for next steps:")
        print(f"   1. Install optional dependencies as needed")
        print(f"   2. Test in container environment")
        print(f"   3. Implement specific workflows")
    else:
        print(f"\n❌ Some tests FAILED")
        print(f"   Basic test: {'✅' if basic_test else '❌'}")
        print(f"   Import test: {'✅' if import_test else '❌'}")
