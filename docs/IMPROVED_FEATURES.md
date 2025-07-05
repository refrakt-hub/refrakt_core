# Improved Refrakt Features

This document describes the new and improved features added to the Refrakt framework.

## Overview

The following improvements have been made to make Refrakt more robust, flexible, and user-friendly:

1. **Safe Registry System** - Thread-safe, import-safe registry replacement
2. **Improved Logging Configuration** - Better logging management with context awareness
3. **Hyperparameter Overrides** - Command-line parameter overrides
4. **Dynamic Dataset Loading** - Support for custom zip files and automatic format detection
5. **Standard Transforms** - Image resizing with size validation
6. **Comprehensive Testing** - Smoke, sanity, and unit tests for all new features

## 1. Safe Registry System

### Overview
The new safe registry system (`src/refrakt_core/registry/safe_registry.py`) provides a drop-in replacement for the existing registry system with improved safety and thread-safety.

### Key Features
- **Thread-safe singleton pattern** - No global variables
- **Import callbacks** - Safe imports with automatic registration
- **Backward compatibility** - Works with existing code
- **Error handling** - Graceful fallbacks and clear error messages

### Usage

```python
from refrakt_core.registry.safe_registry import (
    register_model, get_model, register_dataset, get_dataset
)

# Register a model
@register_model("my_model")
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

# Get the model
model_cls = get_model("my_model")
model = model_cls()
```

### Backward Compatibility
The new registry system maintains backward compatibility with existing code:

```python
# Old way (still works)
from refrakt_core.registry import MODEL_REGISTRY
MODEL_REGISTRY["my_model"] = MyModel

# New way (recommended)
@register_model("my_model")
class MyModel(torch.nn.Module):
    pass
```

## 2. Improved Logging Configuration

### Overview
The improved logging system (`src/refrakt_core/logging_config.py`) provides better logging management with context awareness and thread safety.

### Key Features
- **Thread-safe logging manager** - No global state issues
- **Context-aware logging** - Different loggers for different contexts
- **Flexible configuration** - Console, file, and debug logging
- **Error handling** - Graceful fallbacks

### Usage

```python
from refrakt_core.logging_config import configure_logger, get_logger

# Configure different loggers
train_logger = configure_logger("training", console=True, debug=True)
eval_logger = configure_logger("evaluation", console=True, debug=False)

# Use loggers
train_logger.info("Starting training process")
eval_logger.warning("Model performance below threshold")

# Get logger by name
logger = get_logger("training")
```

## 3. Hyperparameter Overrides

### Overview
The hyperparameter override system (`src/refrakt_core/config/hyperparameter_override.py`) allows command-line overrides of configuration parameters.

### Key Features
- **Command-line overrides** - Override any config parameter
- **Type conversion** - Automatic conversion of strings to appropriate types
- **Validation** - Validate override format and values
- **Integration** - Works with existing configuration system

### Usage

#### Command-line Usage
```bash
# Override model parameters
python -m refrakt_core.api.train \
    config.model.name=ResNet \
    config.optimizer.lr=0.0005 \
    config.trainer.epochs=20

# Override multiple parameters
refrakt -m --config ./path/to/config.yaml \
    config.model.name=ResNet \
    config.optimizer.lr=0.0005 \
    config.trainer.epochs=20
```

#### Programmatic Usage
```python
from refrakt_core.config.hyperparameter_override import apply_overrides

config = {
    "model": {"name": "default", "params": {"lr": 0.001}},
    "training": {"epochs": 10}
}

overrides = [
    "model.name=ResNet",
    "model.params.lr=0.0005",
    "training.epochs=20"
]

result = apply_overrides(config, overrides)
```

#### Supported Value Types
- **Strings**: `config.model.name=ResNet`
- **Numbers**: `config.optimizer.lr=0.001`
- **Booleans**: `config.debug=true`
- **Integers**: `config.epochs=100`

## 4. Dynamic Dataset Loading

### Overview
The dynamic dataset loader (`src/refrakt_core/data/dataset_loader.py`) provides flexible dataset loading with support for custom zip files and automatic format detection.

### Key Features
- **Custom zip file support** - Load datasets from zip files
- **Automatic format detection** - GAN, supervised, contrastive
- **Size validation** - Prevent loading oversized images
- **Torchvision integration** - Works with existing torchvision datasets

### Supported Formats

#### GAN Format
```
dataset.zip/
├── lr/
│   ├── image1.png
│   └── image2.png
└── hr/
    ├── image1.png
    └── image2.png
```

#### Supervised Format
```
dataset.zip/
├── train/
│   ├── cat/
│   │   ├── cat1.png
│   │   └── cat2.png
│   └── dog/
│       ├── dog1.png
│       └── dog2.png
└── val/
    ├── cat/
    └── dog/
```

#### Contrastive Format
```
dataset.zip/
└── images/
    ├── image1.png
    ├── image2.png
    └── image3.png
```

### Usage

```python
from refrakt_core.loaders.dataset_loader import load_dataset

# Load custom dataset
train_dataset, val_dataset = load_dataset("path/to/dataset.zip")

# Load torchvision dataset
train_dataset, val_dataset = load_dataset("mnist")

# Specify dataset type
train_dataset, val_dataset = load_dataset(
    "path/to/dataset.zip", 
    dataset_type="custom"
)
```

## 5. Standard Transforms

### Overview
The standard transforms system (`src/refrakt_core/transforms/standard_transforms.py`) provides image resizing with size validation and multiple resize strategies.

### Key Features
- **Size validation** - Prevent information loss from oversized images
- **Multiple resize strategies** - Maintain aspect ratio, crop, stretch
- **Standard sizes** - Default 224x224 with configurable sizes
- **Tensor support** - Works with both PIL images and tensors

### Usage

```python
from refrakt_core.transforms.standard_transforms import (
    create_standard_transform,
    validate_image_size
)

# Create transform
transform = create_standard_transform(
    target_size=(224, 224),
    resize_strategy="maintain_aspect",
    normalize=True,
    augment=False
)

# Validate image size
is_valid, error_msg = validate_image_size(
    image_path, 
    max_size=(448, 448)
)

# Apply transform
transformed_image = transform(image)
```

### Resize Strategies
- **maintain_aspect**: Resize maintaining aspect ratio with padding
- **crop**: Resize and crop to target size
- **stretch**: Stretch to target size (may distort)

## 6. Integration with Existing Pipeline

### Model Builder Integration
The model builder has been updated to support hyperparameter overrides:

```python
from refrakt_core.api.builders.model_builder import build_model

# Build model with overrides
model = build_model(
    cfg=config,
    modules=modules,
    device="cuda",
    overrides=["model.params.lr=0.0005", "model.name=ResNet"]
)
```

### Command-line Integration
The system integrates with command-line tools:

```bash
# Extract overrides from command-line arguments
python train.py --config config.yaml model.name=ResNet optimizer.lr=0.001

# The system automatically:
# 1. Extracts overrides: ["model.name=ResNet", "optimizer.lr=0.001"]
# 2. Applies them to the configuration
# 3. Builds the model with overridden parameters
```

## 7. Testing

### Test Structure
All new features include comprehensive tests:

- **Smoke tests** - Basic functionality verification
- **Sanity tests** - Edge case handling
- **Unit tests** - Individual component testing
- **Integration tests** - System-wide testing

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/config/  # Hyperparameter override tests
pytest tests/registry/  # Registry tests
pytest tests/data/  # Dataset loader tests
pytest tests/transforms/  # Transform tests
```

## 8. Migration Guide

### From Old Registry to Safe Registry
```python
# Old way
from refrakt_core.registry import MODEL_REGISTRY
MODEL_REGISTRY["my_model"] = MyModel

# New way (recommended)
from refrakt_core.registry.safe_registry import register_model
@register_model("my_model")
class MyModel(torch.nn.Module):
    pass
```

### From Global Logging to Improved Logging
```python
# Old way
from refrakt_core.global_logging import get_global_logger
logger = get_global_logger()

# New way (recommended)
from refrakt_core.logging_config import configure_logger
logger = configure_logger("my_module", console=True)
```

### Adding Hyperparameter Overrides
```python
# Before
config = load_config("config.yaml")
model = build_model(config, modules, device)

# After
config = load_config("config.yaml")
overrides = extract_overrides_from_args(sys.argv[1:])
model = build_model(config, modules, device, overrides=overrides)
```

## 9. Examples

### Complete Example
See `examples/improved_features_demo.py` for a complete demonstration of all features.

### Configuration Example
```yaml
# config.yaml
model:
  name: ResNet
  params:
    input_size: 784
    output_size: 10
    lr: 0.001

training:
  epochs: 10
  batch_size: 32

data:
  dataset: mnist
  transform: standard
```

### Command-line Override
```bash
python train.py \
  --config config.yaml \
  model.params.lr=0.0005 \
  training.epochs=20 \
  data.dataset=cifar10
```

## 10. Best Practices

### Registry Usage
1. Use decorators for registration when possible
2. Provide meaningful names for components
3. Handle missing components gracefully
4. Use the safe registry for new code

### Logging
1. Use context-specific loggers
2. Set appropriate log levels
3. Include relevant context in log messages
4. Handle logging errors gracefully

### Hyperparameter Overrides
1. Use descriptive parameter names
2. Validate override values
3. Provide helpful error messages
4. Document supported override formats

### Dataset Loading
1. Validate dataset structure before loading
2. Use appropriate transforms for your task
3. Handle missing or corrupted data gracefully
4. Provide clear error messages for format issues

### Transforms
1. Validate image sizes before processing
2. Choose appropriate resize strategies
3. Handle different image formats
4. Provide fallbacks for edge cases

## Conclusion

These improvements make Refrakt more robust, flexible, and user-friendly while maintaining backward compatibility. The new features provide better error handling, more flexible configuration, and improved developer experience. 