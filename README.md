# 🌳 Forestry Carbon ARR Library

A Python library for managing **GEE_notebook_Forestry** integration in forestry carbon analysis workflows. This library provides import and path management for integrating with GEE_notebook_Forestry, supporting both local development (side-by-side) and container environments.

## ✨ Features

- 🌍 **Flexible Integration**: Works with or without GEE_notebook_Forestry
- 🔧 **Container Ready**: Optimized for Docker and containerized deployments
- 📁 **Path Management**: Automatic detection and resolution of GEE_notebook_Forestry paths
- 🐍 **Import Strategy**: Smart import management for local vs container environments
- ⚙️ **Configuration**: Flexible configuration management
- 🎯 **Production Ready**: Designed for real-world deployment scenarios

## 🚀 Quick Start

### Prerequisites

**IMPORTANT**: For full functionality, you need to clone `GEE_notebook_Forestry` alongside `forestry_carbon_arr`:

```bash
# Clone both repositories side-by-side
git clone <forestry_carbon_arr_repo>
git clone <GEE_notebook_Forestry_repo>

# Your directory structure should look like:
your-project/
├── forestry_carbon_arr/          # This library
│   ├── __init__.py
│   ├── core/
│   └── utils/
└── GEE_notebook_Forestry/        # Required for full functionality
    ├── osi/
    │   ├── __init__.py
    │   ├── image_collection/
    │   ├── fcd/
    │   └── ml/
    └── notebooks/
```

### Installation

```bash
# Basic installation
pip install forestry_carbon_arr

# With Google Earth Engine support (optional)
pip install forestry_carbon_arr[gee]
```

### Basic Usage

```python
from forestry_carbon_arr import ForestryCarbonARR

# Initialize the system (auto-detects GEE_notebook_Forestry if side-by-side)
forestry = ForestryCarbonARR()

# Check system status
print(forestry.get_system_info())

# Check if GEE_notebook_Forestry integration is working
if forestry.gee_forestry_available:
    print("✅ GEE_notebook_Forestry integration is working!")
    print(f"Import strategy: {forestry.import_strategy}")
    print(f"Path: {forestry.gee_forestry_path}")
else:
    print("⚠️  GEE_notebook_Forestry not found - integration not available")

# Get import guide
guide = forestry.get_import_guide()
print(guide)
```

## 🏗️ Architecture

### Flexible Dependency Management

The library is designed to work in multiple scenarios:

1. **With GEE_notebook_Forestry (Side-by-Side)**: Full functionality with advanced features
2. **With GEE_notebook_Forestry (Container)**: Full functionality in Docker environments
3. **Standalone**: Core functionality without external dependencies
4. **Development**: Flexible for research and development workflows

### Integration Strategy

#### Local Development (Side-by-Side)
```python
# Directory structure:
your-project/
├── forestry_carbon_arr/
└── GEE_notebook_Forestry/

# Automatic detection and integration
forestry = ForestryCarbonARR()  # Auto-detects side-by-side GEE_notebook_Forestry

# Import examples:
from GEE_notebook_Forestry.osi import *
from GEE_notebook_Forestry.osi.image_collection.main import ImageCollection
```

#### Container Environment
```python
# Container paths (automatic detection):
# /usr/src/app/gee_lib (development)
# /app/gee_lib (production)

# Import examples:
from gee_lib.osi import *
from gee_lib.osi.image_collection.main import ImageCollection
```

#### Manual Path Specification
```python
# Manual path specification
forestry = ForestryCarbonARR(gee_forestry_path="/path/to/GEE_notebook_Forestry")
```

## 📋 Configuration

### Configuration File Example

```json
{
    "project": {
        "name": "carbon_project_2024",
        "region": "southeast_asia",
        "description": "ARR Carbon Project Analysis"
    },
    "gee": {
        "project_id": "your-gee-project-id",
        "initialize": true
    },
    "satellite": {
        "provider": "Sentinel",
        "date_range": ["2024-01-01", "2024-12-31"],
        "cloud_cover_threshold": 40
    },
    "ml": {
        "algorithm": "gbm",
        "training_samples": 1000
    }
}
```

### Environment Variables

```bash
# GEE Forestry path (optional)
export GEE_FORESTRY_PATH="/path/to/GEE_notebook_Forestry"

# GEE project ID
export GEE_PROJECT_ID="your-gee-project-id"
```

## 🔧 Container Integration

### Docker Compose Integration

The library is designed to work seamlessly with your existing container setup:

```yaml
# docker-compose.yml
services:
  jupyter:
    volumes:
      - ./forestry_carbon_arr:/usr/src/app/forestry_carbon_arr
      - ./GEE_notebook_Forestry:/usr/src/app/gee_lib:ro
    environment:
      - PYTHONPATH=/usr/src/app:/usr/src/app/gee_lib:/usr/src/app/forestry_carbon_arr
```

### Container Paths

The library automatically detects and uses these container paths:
- `/usr/src/app/gee_lib` (GEE_notebook_Forestry)
- `/usr/src/app/forestry_carbon_arr` (This library)
- `/usr/src/app/ex_ante_lib` (Ex-ante library)

## 📁 Setup Requirements

### Required Directory Structure

For full functionality, you **MUST** clone `GEE_notebook_Forestry` alongside `forestry_carbon_arr`:

```
your-project/
├── forestry_carbon_arr/          # This library
│   ├── __init__.py
│   ├── core/
│   ├── utils/
│   ├── config/
│   └── README.md
└── GEE_notebook_Forestry/        # REQUIRED for full functionality
    ├── osi/
    │   ├── __init__.py
    │   ├── image_collection/
    │   │   └── main.py
    │   ├── fcd/
    │   │   └── main_fcd.py
    │   ├── ml/
    │   │   └── main.py
    │   └── utils/
    │       └── main.py
    └── notebooks/
```

### Setup Commands

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
python -c "from forestry_carbon_arr import ForestryCarbonARR; print('✅ Setup complete!')"
```

### Verification

Run the setup verification:

```bash
cd forestry_carbon_arr
python test_import_strategies.py
```

Expected output:
```
✅ GEE_notebook_Forestry found side-by-side
✅ osi directory found
✅ Import strategy: local
🎉 All tests PASSED!
```

## 📚 Usage Examples

### Basic Carbon Analysis

```python
from forestry_carbon_arr import ForestryCarbonARR
import geopandas as gpd

# Initialize
forestry = ForestryCarbonARR()

# Load AOI
aoi = gpd.read_file("path/to/aoi.shp")

# Create pipeline
pipeline = forestry.create_analysis_pipeline({
    'project': {'name': 'test_project'},
    'satellite': {'provider': 'Sentinel'}
})

# Run analysis
results = pipeline.run_analysis(aoi)
```

### Advanced GEE Integration

```python
# With GEE_notebook_Forestry integration
forestry = ForestryCarbonARR()

# Access GEE processor
gee_processor = forestry.gee_processor

# Create image collection
collection = gee_processor.create_image_collection(
    satellite_provider="Sentinel",
    date_range=["2024-01-01", "2024-12-31"],
    aoi=aoi_geometry
)

# Calculate FCD
fcd_results = gee_processor.calculate_forest_canopy_density(image)
```

### Machine Learning Analysis

```python
# ML analysis
ml_analyzer = forestry.ml_analyzer

# Load training data
training_data = gpd.read_file("path/to/training.shp")

# Perform classification
classified = ml_analyzer.classify_landcover(
    image=satellite_image,
    training_data=training_data,
    algorithm="gbm"
)
```

## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/forestry_carbon_arr.git
cd forestry_carbon_arr

# Install in development mode
pip install -e .[dev]

# Run tests
pytest

# Format code
black .
flake8 .
```

### Adding New Features

1. Create feature branch
2. Implement feature with tests
3. Update documentation
4. Submit pull request

## 📊 System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 10GB disk space

### Recommended Requirements
- Python 3.10+
- 8GB+ RAM
- 50GB+ disk space
- Google Earth Engine account

### Optional Dependencies
- GEE_notebook_Forestry (for advanced features)
- ArcGIS Pro (for GIS integration)
- GPU support (for ML acceleration)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest tests/

# Check code quality
black .
flake8 .
mypy .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Google Earth Engine](https://earthengine.google.com/) for satellite data processing
- [GEE_notebook_Forestry](https://github.com/yourusername/GEE_notebook_Forestry) for advanced forestry analysis
- The open-source geospatial community
- Carbon project developers and researchers

## 📞 Support

- 📧 Email: muh.firdausiqbal@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/forestry_carbon_arr/issues)
- 📖 Documentation: [Wiki](https://github.com/yourusername/forestry_carbon_arr/wiki)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/forestry_carbon_arr/discussions)

---

**Made with ❤️ for carbon project developers and forestry researchers**