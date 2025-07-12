# About

**refrakt_core** is a modular deep learning and machine learning research framework for computer vision, designed for rapid experimentation, extensibility, and reproducibility. It now features a robust, thread-safe registry system, dynamic dataset handling, advanced image resizing, flexible hyperparameter overrides, and comprehensive logging and testing. Refrakt supports both classic and modern CV/ML papers, and enables seamless ML/DL/fusion pipelines.

> This project aims to unify, extend, and visualize foundational and modern architectures through clean code, clear abstractions, and rigorous logging.

## 🚀 Key Features

- **Safe Registry System**: Thread-safe, import-safe, decorator-based registration for models, datasets, losses, trainers, and transforms. Backward compatible with legacy code.
- **Dynamic Dataset Loader**: Load datasets from custom zip files or torchvision, with automatic format detection (GAN, supervised, contrastive) and size validation.
- **Standard Image Resizer/Transforms**: Multiple resize strategies (maintain aspect, crop, stretch), size validation, and tensor/PIL support.
- **Hyperparameter Overrides**: Override any config parameter from the command line or programmatically for fast experimentation.
- **Improved Logging**: Context-aware logging with better error handling, supporting both TensorBoard and Weights & Biases (W&B).
- **Comprehensive Testing**: Smoke, sanity, unit, and integration tests for all major features.
- **ML/DL/Fusion Pipelines**: Support for pure-ML, pure-DL, and hybrid fusion pipelines (e.g., deep feature extraction + ML fusion head).
- **Modular YAML Configs**: All components (model, trainer, loss, optimizer, scheduler, feature engineering) are defined in modular YAML files.

## 📚 Implemented Papers

- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) – *An Image is Worth 16x16 Words*
- [ResNet](https://arxiv.org/abs/1512.03385) – *Deep Residual Learning for Image Recognition*
- [Autoencoders](https://www.cs.toronto.edu/~hinton/science.pdf) – *Learning Representations via Reconstruction*
- [Swin Transformer](https://arxiv.org/abs/2103.14030) – *Hierarchical Vision Transformer with Shifted Windows*
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [ConvNeXt](https://arxiv.org/abs/2201.03545) – *A ConvNet for the 2020s*
- [SRGAN](https://arxiv.org/abs/1609.04802) – *Photo-Realistic Single Image Super-Resolution with GANs*
- [SimCLR](https://arxiv.org/abs/2002.05709) – *A Simple Framework for Contrastive Learning*
- [DINO](https://arxiv.org/abs/2104.14294) – *Self-Supervised Vision Transformers*
- [MAE](https://arxiv.org/abs/2111.06377) – *Masked Autoencoders*
- [MSN](https://arxiv.org/abs/2204.07141) – *Masked Siamese Networks*

## ⚙️ Setup
```bash
# For pip install 
pip install refrakt_core
```

```bash
# Manual setup
git clone https://github.com/refrakt-hub/refrakt_core.git
cd refrakt_core

# Create and activate a virtual environment
conda create -n refrakt python=3.10 -y
conda activate refrakt

# Install dependencies
pip install -r requirements.txt
```

### GPU/cuML Support

If you want to use GPU-accelerated ML features (cuML), you must manually install the required dependencies after the main install. Run one of the following scripts from the project root:

```bash
# For bash users:
./install_cuml.sh

# For fish shell users:
./install_cuml.fish
```

This will install the appropriate cuML and RAPIDS libraries for your environment. If you do not need GPU/cuML support, you can skip this step.

## 🧪 Running Experiments

```bash
# Run with a config file
python -m refrakt_core.api --config refrakt_core/config/vit.yaml

# Or using the CLI
refrakt --config ./refrakt_core/config/resnet.yaml

# Override hyperparameters on-the-fly
python -m refrakt_core.api.train \
    config.optimizer.lr=0.0005 \
    config.trainer.epochs=20
```

### Supported CLI Flags

| Flag         | Description                                              |
| ------------ | -------------------------------------------------------- |
| `--config`   | Path to YAML config file                                 |
| `--log_type` | Logging backend: `tensorboard`, `wandb`, or both         |
| `--debug`    | Enable debug mode with extra verbosity                   |

## 🔧 Config Structure (YAML)

All components are defined in modular YAML files under `refrakt_core/config/`.

```yaml
runtime:
  mode: pipeline
  log_type: []

dataset:
  name: MNIST
  params:
    root: ./data
    train: true
    download: true
  transform:
    - name: Resize
      params: { size: [28, 28] }
    - name: ToTensor
    - name: Normalize
      params:
        mean: [0.1307]
        std: [0.3081]

dataloader:
  params:
    batch_size: 32
    shuffle: true
    num_workers: 4
    drop_last: false

model:
  name: vit
  wrapper: vit
  params:
    in_channels: 1
    num_classes: 10
    image_size: 28 
    patch_size: 7
  fusion:
    type: cuml
    model: logistic_regression
    params:
      C: 1.0
      penalty: l2
      solver: qn
      max_iter: 1000

loss:
  name: ce_wrapped
  mode: logits
  params: {}

optimizer:
  name: adamw
  params:
    lr: 0.0003

scheduler: null

trainer:
  name: supervised
  params:
    save_dir: "./checkpoints"
    num_epochs: 1
    device: cuda
```

## 🧩 Major Components & Patterns

### 1. Safe Registry System

Register models, datasets, losses, trainers, and transforms using decorators:

```python
from refrakt_core.registry.safe_registry import register_model, get_model

@register_model("my_model")
class MyModel(torch.nn.Module):
    ...

model_cls = get_model("my_model")
model = model_cls()
```

### 2. Dynamic Dataset Loader

Load datasets from zip files or torchvision, with format detection:

```python
from refrakt_core.loaders.dataset_loader import load_dataset
train_dataset, val_dataset = load_dataset("path/to/dataset.zip")
train_dataset, val_dataset = load_dataset("mnist")
```

### 3. Standard Image Resizer/Transforms

```python
from refrakt_core.resizers.standard_transforms import create_standard_transform
transform = create_standard_transform(target_size=(224, 224), resize_strategy="maintain_aspect")
```

### 4. Hyperparameter Overrides

Override any config value from the command line or programmatically:

```bash
python train.py --config config.yaml model.name=ResNet optimizer.lr=0.001
```

### 5. ML/DL/Fusion Pipelines

Supports pure-ML, pure-DL, and hybrid fusion pipelines (deep features + ML head):

```python
from refrakt_core.api.builders.model_builder import build_model
model = build_model(cfg=config, modules=modules, device="cuda", overrides=["model.params.lr=0.0005"])
```

## 📈 Logging & Monitoring

- **TensorBoard**: logs in `logs/<model_name>/tensorboard/`
- **Weights & Biases**: auto-logged if enabled in config

```bash
tensorboard --logdir=./logs/<model_name>/tensorboard/
export WANDB_API_KEY=your_key_here
```

## 🧱 Project Structure

```
refrakt_core/
├── api/                  # CLI: train.py, test.py, inference.py
│   └── builders/         # Builders for models, losses, optimizers, datasets
├── config/               # YAML configurations for each experiment
├── losses/               # Contrastive, GAN, MAE, VAE, etc.
├── models/               # Vision architectures (ViT, ResNet, MAE, etc.)
│   └── templates/        # Base model templates and abstractions
├── trainer/              # Task-specific training logic (SimCLR, SRGAN, etc.)
├── registry/             # Safe, decorator-based plugin system
├── utils/                # Helper modules (encoders, decoders, data classes)
├── resizers/             # Image resizing and standard transforms
├── loaders/              # Dynamic and standard dataset loaders
├── transforms.py         # Data augmentation logic
├── datasets.py           # Dataset definitions and loader helpers
├── logging_config.py     # Logger wrapper for stdout + W&B/TensorBoard
```

## 🧪 Testing

Run all tests:
```bash
pytest tests/
```

## 🧩 Extending Refrakt

### Add a New Model

1. Create the architecture in `models/your_model.py`
2. Inherit from a base class in `models/templates/models.py`
3. Register it using:

```python
from refrakt_core.registry.model_registry import register_model

@register_model("your_model")
class YourModel(BaseClassifier):
    ...
```

4. Add a YAML config: `config/your_model.yaml`
5. Write a custom trainer if needed (`trainer/your_model.py`)

### Add a Custom Dataset Loader or Transform
- Implement in `loaders/` or `resizers/`
- Register with the safe registry

## 🔍 Example Output

- Progress bar (via `tqdm`)
- Metrics printed and logged
- `./logs/<model_name>/` with TensorBoard events
- W&B dashboard if enabled

## 📬 Contributing

1. Clone and install:
   ```bash
   git clone ...
   pip install -r requirements-dev.txt
   pre-commit install
   ```
2. Follow formatting (`black`, `isort`, `pylint`)
3. Write tests for any new feature
4. Run:
   ```bash
   pytest tests/
   ```

> PRs and issues are welcome!

## 🔭 Future Scope

| Milestone  | Description                                             |
| ---------- | ------------------------------------------------------- |
| ✅ Stage 1  | Paper re-implementations in notebooks                   |
| ✅ Stage 2  | Modular training + model pipelines                      |
| ✅ Stage 3  | Python library (`refrakt train`, etc.)                  |
| 🔜 Stage 4 | TBD |

Planned additions:
- Much better code readability + extensive documentation (`readthedocs`)
- More sklearn and cuML models made available through the registry. 
- Integration of Kolmogorov-Arnold Networks and Lagrangian Neural Networks.
- Checkpoints for pre-trained weights of models saved. 
- Integrate model tracing for Fusion Blocks. 
- Allow for generative / latent fusion trainng. 

## 📄 License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for full details.

## 👤 Maintainer

**Akshath Mangudi**
If you find issues, raise them. If you learn from this, share it.
Built with love and curiosity :)

## 🤝 Contributing

We welcome contributions! To get started:

- See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines, including development setup, code style, and testing.
- Set up your dev environment with:
  ```bash
  pip install -e .[dev]
  # or
  python scripts/dev_setup.py
  ```
- This will install all runtime and development dependencies (testing, linting, formatting, type checking, etc.) and set up pre-commit hooks for code quality.
- Please ensure your code passes all pre-commit checks and tests before opening a pull request.

---
