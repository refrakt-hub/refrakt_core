# About

**refrakt_core** is a deep learning research framework that re-implements key computer vision papers with high modularity, extensibility, and reproducibility. Designed for education, experimentation, and benchmarking, Refrakt enables rapid iteration on architectures, loss functions, and training routines — all driven via configuration.

> This project is part of a broader initiative to understand, unify, and visualize foundational and modern architectures through clean code, clear abstractions, and rigorous logging.

## 📌 Goals

- Faithfully reproduce influential CV papers
- Modularize training/evaluation pipelines using YAML + registry patterns
- Enable quick debugging and metric logging via TensorBoard and Weights & Biases (W&B)
- Serve as a platform for personal research, fine-tuning, or new model design

## 📚 Implemented Papers

- **Vision Transformer (ViT)** – *An Image is Worth 16x16 Words*
- **ResNet** – *Deep Residual Learning for Image Recognition*
- **Autoencoders** – *Learning Representations via Reconstruction*
- **Swin Transformer** – *Hierarchical Vision Transformer with Shifted Windows*
- **Attention is All You Need**
- **ConvNeXt** – *A ConvNet for the 2020s*
- **SRGAN** – *Photo-Realistic Single Image Super-Resolution with GANs*
- **SimCLR** – *A Simple Framework for Contrastive Learning*
- **DINO** – *Self-Supervised Vision Transformers*
- **MAE** – *Masked Autoencoders*
- **MSN** – *Masked Siamese Networks*


## ⚙️ Setup

```bash
git clone https://github.com/refrakt-hub/refrakt_core.git
cd refrakt_core

# Create and activate a virtual environment
conda create -n refrakt python=3.10 -y
conda activate refrakt

# Install dependencies
pip install -r requirements.txt
````


## 🧪 Running Experiments

```bash
# Calling __main__.py
python -m refrakt_core.api --config refrakt_core/config/vit.yaml

# Calling `refrakt` as a command. 
refrakt --config ./refrakt_core/config/resnet.yaml


# Override on-the-fly
python -m refrakt_core.api.train \
    config.model.name=ResNet \
    config.optimizer.lr=0.0005 \
    config.trainer.epochs=20
```

## CLI Usage
```bash
usage: refrakt [-h] --config CONFIG --mode {train,test,inference,pipeline}
               [--log_dir LOG_DIR]
               [--log_type [{tensorboard,wandb} ...]]
               [--console]
               [--model_path MODEL_PATH]
               [--debug]
```

| Flag       | Description                                                 |
| ---------- | ----------------------------------------------------------- |
| `--config` | Path to YAML config file                                    |
| `--mode`   | Execution mode: `train`, `test`, `inference`, or `pipeline` |
| `--log_dir`    | Path to log directory (e.g., `./runs`)              |
| `--log_type`   | Logging backend: `tensorboard`, `wandb`, or both    |
| `--console`    | Print logs to console as well                       |
| `--model_path` | Path to model weights/checkpoint for inference/test |
| `--debug`      | Enable debug mode with extra verbosity              |


## 🔧 Config Structure (YAML)

All components (model, trainer, loss, optimizer, scheduler) are defined in modular YAML files under `refrakt_core/config/`.

### Example: `vit.yaml`

```yaml
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
  params:
    in_channels: 1
    num_classes: 10
    image_size: 28 
    patch_size: 7

loss:
  name: cross_entropy
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
    model_name: "vit"
    num_epochs: 1
    device: cuda
```

## 📈 Logging & Monitoring

### ✅ Supported
* **TensorBoard** (logs saved in `runs/`)
* **Weights & Biases** (auto-logged via config)

### Setup

```bash
# TensorBoard
tensorboard --logdir=./logs/<model_name>/tensorboard/

# Weights & Biases
export WANDB_API_KEY=your_key_here
```

## 🧱 Project Structure

```
refrakt_core/
├── api/                  # CLI: train.py, test.py, inference.py
│   └── builders/         # Builders for models, losses, optimizers, datasets
│
├── config/               # YAML configurations for each experiment
│   └── schema.py         # Pydantic validation for configs
│
├── losses/               # Contrastive, GAN, MAE, VAE, etc.
│
├── models/               # Vision architectures (ViT, ResNet, MAE, etc.)
│   └── templates/        # Base model templates and abstractions
│
├── trainer/              # Task-specific training logic (SimCLR, SRGAN, etc.)
│
├── registry/             # Decorator-based plugin system
│
├── utils/                # Helper modules (encoders, decoders, data classes)
│
├── transforms.py         # Data augmentation logic
├── datasets.py           # Dataset definitions and loader helpers
├── logging.py            # Logger wrapper for stdout + W&B/TensorBoard
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
5. Write a custom trainer if needed (`trainer/your_model.py`)s

## 🔍 Example Output

Once training begins, expect:

* Printed progress bar (via `tqdm`)
* Metrics like loss/accuracy printed and logged
* `./logs/<model_name>/` with `events.out.tfevents` for TensorBoard
* W\&B dashboard with training curves, if enabled


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

> PRs and creating issues are absolutely welcome.

## 🔭 Future Scope

| Milestone  | Description                                             |
| ---------- | ------------------------------------------------------- |
| ✅ Stage 1  | Paper re-implementations in notebooks                   |
| ✅ Stage 2  | Modular training + model pipelines                      |
| 🔜 Stage 3 | Convert to a Python library (`refrakt train`, etc.)     |
| 🔜 Stage 4 | Web app: visual dashboards, interactive inference       |
| 🔜 Stage 5 | No-code/low-code config engine for multimodal pipelines |
| 🔜 Stage 6 | Explainability: saliency maps, attention visualizations |

Planned additions include:

* Mixed-modality support (e.g., image + text)
* More contrastive/self-supervised frameworks
* System-level benchmarks across GPUs
* Evaluation on downstream tasks (detection, segmentation)

## 📄 License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for full details.

## 👤 Maintainer

**Akshath Mangudi**
If you find issues, raise them. If you learn from this, share it.
Built with love and curiosity :)
