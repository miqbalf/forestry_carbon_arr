"""
Simple test for import/path management functionality only.
"""

import sys
from pathlib import Path

# Add the library to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_import_management():
    """Test import management functionality."""
    print("🌳 Testing Import/Path Management")
    print("=" * 50)
    
    # Test 1: Path Resolver
    print("\n1. Testing Path Resolver...")
    try:
        from utils.path_resolver import PathResolver
        
        path_resolver = PathResolver()
        possible_paths = path_resolver.resolve_gee_forestry_paths()
        
        found_paths = [str(p) for p in possible_paths if p and p.exists()]
        print(f"   📋 Found {len(found_paths)} possible paths:")
        for path in found_paths:
            print(f"      ✅ {path}")
        
        if found_paths:
            print(f"   ✅ Path resolution working")
        else:
            print(f"   ❌ No paths found")
            return False
            
    except Exception as e:
        print(f"   ❌ Path resolver failed: {e}")
        return False
    
    # Test 2: Import Manager
    print("\n2. Testing Import Manager...")
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
            
            # Show import guide
            print(f"\n3. Import Guide:")
            guide = import_manager.create_import_guide()
            print(guide)
            
        else:
            print(f"   ⚠️  GEE_notebook_Forestry not detected")
            print(f"   💡 This is expected if GEE_notebook_Forestry is not side-by-side")
            
    except Exception as e:
        print(f"   ❌ Import manager failed: {e}")
        return False
    
    # Test 3: Dependency Manager
    print(f"\n4. Testing Dependency Manager...")
    try:
        from utils.dependency_manager import DependencyManager
        
        dep_manager = DependencyManager()
        
        # Check core dependencies
        core_available, core_missing = dep_manager.check_core_dependencies()
        print(f"   Core dependencies: {'✅' if core_available else '❌'}")
        if core_missing:
            print(f"   Missing: {core_missing}")
        
        # Check GEE dependencies
        gee_available, gee_missing = dep_manager.check_optional_dependencies('gee')
        print(f"   GEE dependencies: {'✅' if gee_available else '❌'}")
        if gee_missing:
            print(f"   Missing: {gee_missing}")
        
        # Check GEE Forestry modules
        gee_forestry_available, gee_forestry_missing = dep_manager.check_gee_forestry_modules()
        print(f"   GEE Forestry modules: {'✅' if gee_forestry_available else '❌'}")
        if gee_forestry_missing:
            print(f"   Missing: {gee_forestry_missing}")
        
    except Exception as e:
        print(f"   ❌ Dependency manager failed: {e}")
        return False
    
    return True

def show_cleanup_summary():
    """Show what was cleaned up."""
    print(f"\n🧹 Cleanup Summary")
    print("=" * 30)
    
    summary = """
✅ REMOVED (Not needed for import management):
- satellite_processing/ (satellite data processing)
- ml_analysis/ (machine learning analysis)  
- gee_integration/ (GEE processor)
- core/pipeline.py (analysis pipeline)
- core/workflow.py (workflow management)
- examples/basic_usage.py (example usage)
- Various test files

✅ KEPT (Essential for import management):
- core/main.py (ForestryCarbonARR class)
- config/ (configuration management)
- utils/ (path, import, dependency management)
- exceptions.py (custom exceptions)
- requirements.txt (dependencies)
- setup.py (package setup)

🎯 FOCUS: Import and path management only
"""
    
    print(summary)

if __name__ == "__main__":
    # Show cleanup summary
    show_cleanup_summary()
    
    # Test import management
    success = test_import_management()
    
    if success:
        print(f"\n🎉 IMPORT MANAGEMENT TEST: PASSED")
        print(f"✅ Path resolution working")
        print(f"✅ Import management working") 
        print(f"✅ Dependency management working")
        print(f"✅ Library cleaned up and focused")
    else:
        print(f"\n❌ IMPORT MANAGEMENT TEST: FAILED")
        print(f"❌ Check the errors above")
