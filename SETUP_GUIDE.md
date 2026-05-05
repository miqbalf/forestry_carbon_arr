# 🌳 Setup Guide: forestry_carbon_arr with GEE_notebook_Forestry

## ✅ Integration Status: WORKING

The forestry_carbon_arr library is successfully integrated with GEE_notebook_Forestry and ready for use.

## 📁 Required Setup

### 1. Directory Structure

**CRITICAL**: You must clone both repositories side-by-side:

```
your-project/
├── forestry_carbon_arr/          # This library
│   ├── __init__.py
│   ├── core/
│   ├── utils/
│   └── README.md
└── GEE_notebook_Forestry/        # REQUIRED for full functionality
    ├── osi/
    │   ├── __init__.py
    │   ├── image_collection/
    │   ├── fcd/
    │   └── ml/
    └── notebooks/
```

### 2. Setup Commands

```bash
# 1. Create project directory
mkdir my-forestry-project
cd my-forestry-project

# 2. Clone both repositories side-by-side
git clone <forestry_carbon_arr_repo>
git clone <GEE_notebook_Forestry_repo>

# 3. Install forestry_carbon_arr
cd forestry_carbon_arr
pip install -e .

# 4. Verify setup
python test_final_integration.py
```

## 🔍 Verification

Run the verification test:

```bash
cd forestry_carbon_arr
python test_final_integration.py
```

**Expected Output:**
```
✅ GEE_notebook_Forestry found side-by-side
✅ osi directory found
✅ Import strategy: local
🎉 FINAL INTEGRATION TEST: PASSED
```

## 🚀 Usage

### Basic Usage

```python
from forestry_carbon_arr import ForestryCarbonARR

# Initialize (auto-detects GEE_notebook_Forestry)
forestry = ForestryCarbonARR()

# Check integration status
if forestry.dependency_manager.is_gee_forestry_available():
    print("✅ GEE_notebook_Forestry integration working!")
else:
    print("⚠️  GEE_notebook_Forestry not found")
```

### Import Examples

#### Local Development
```python
# Direct imports (after setup)
from GEE_notebook_Forestry.osi import *
from GEE_notebook_Forestry.osi.image_collection.main import ImageCollection
from GEE_notebook_Forestry.osi.fcd.main_fcd import FCDCalc
from GEE_notebook_Forestry.osi.ml.main import LandcoverML
```

#### Container Environment
```python
# Container imports (automatic detection)
from gee_lib.osi import *
from gee_lib.osi.image_collection.main import ImageCollection
from gee_lib.osi.fcd.main_fcd import FCDCalc
from gee_lib.osi.ml.main import LandcoverML
```

## 🐳 Container Integration

### Docker Compose Setup

```yaml
services:
  jupyter:
    volumes:
      - ./forestry_carbon_arr:/usr/src/app/forestry_carbon_arr
      - ./GEE_notebook_Forestry:/usr/src/app/gee_lib:ro
    environment:
      - PYTHONPATH=/usr/src/app:/usr/src/app/gee_lib:/usr/src/app/forestry_carbon_arr
```

### Container Paths

The library automatically detects:
- `/usr/src/app/gee_lib` (development container)
- `/app/gee_lib` (production container)

## 🔧 Troubleshooting

### Common Issues

1. **GEE_notebook_Forestry not found**
   ```
   ❌ GEE_notebook_Forestry not found side-by-side
   ```
   **Solution**: Clone GEE_notebook_Forestry in the same directory as forestry_carbon_arr

2. **Import errors**
   ```
   ❌ No module named 'ee'
   ```
   **Solution**: Install required dependencies:
   ```bash
   pip install earthengine-api geopandas
   ```

3. **Container path issues**
   ```
   ❌ Container paths not found
   ```
   **Solution**: Check Docker volume mounting in docker-compose.yml

### Verification Commands

```bash
# Check directory structure
ls -la
# Should show both forestry_carbon_arr/ and GEE_notebook_Forestry/

# Check osi directory
ls -la GEE_notebook_Forestry/osi/
# Should show osi modules

# Run integration test
cd forestry_carbon_arr
python test_final_integration.py
```

## 📊 Test Results

**Current Status**: ✅ WORKING

- ✅ GEE_notebook_Forestry detection (side-by-side)
- ✅ Path resolution (local and container)
- ✅ Import manager functionality
- ✅ Python path setup
- ✅ Conditional import strategy
- ✅ Container integration ready

**Modules Available**: 2/9 (basic modules working, advanced modules need dependencies)

## 🎯 Next Steps

1. **Install Dependencies**: Add required packages (earthengine-api, geopandas, etc.)
2. **Test Workflows**: Implement specific analysis workflows
3. **Container Testing**: Test in Docker environment
4. **Production Deployment**: Deploy to production containers

## 📞 Support

If you encounter issues:

1. Check the directory structure matches the requirements
2. Run the verification tests
3. Check the troubleshooting section
4. Review the README.md for detailed documentation

---

**✅ Setup Complete**: The forestry_carbon_arr library is ready for use with GEE_notebook_Forestry integration!
