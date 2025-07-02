"""The test module for Refrakt."""

import glob
import os
import traceback
import warnings
from datetime import datetime
from typing import Optional

import torch
from omegaconf import OmegaConf

from refrakt_core.api.builders.dataloader_builder import build_dataloader
from refrakt_core.api.builders.dataset_builder import build_dataset
from refrakt_core.api.builders.loss_builder import build_loss
from refrakt_core.api.builders.model_builder import build_model
from refrakt_core.api.builders.trainer_builder import initialize_trainer
from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.utils.test_utils import (_build_test_loader,
                                               _load_config,
                                               _load_model_checkpoint)
from refrakt_core.global_logging import get_global_logger
from refrakt_core.integrations.fusion.builder import build_fusion_head
from refrakt_core.integrations.sklearn.trainer import FusionTrainer
from refrakt_core.integrations.sklearn.wrapper import SklearnWrapper
from refrakt_core.integrations.cuml.wrapper import CuMLWrapper
from refrakt_core.registry.loss_registry import get_loss
from refrakt_core.registry.model_registry import get_model
from refrakt_core.registry.trainer_registry import get_trainer
from refrakt_core.registry.wrapper_registry import get_wrapper
from refrakt_core.schema.artifact import ArtifactDumper
from refrakt_core.schema.model_output import ModelOutput

warnings.filterwarnings("ignore")


def test(cfg, model_path: Optional[str] = None, logger=None):
    try:
        # === Load config and logger ===
        config = _load_config(cfg)
        runtime_cfg = config.get("runtime", {})
        log_types = runtime_cfg.get("log_type", [])
        log_dir = runtime_cfg.get("log_dir", "./logs")
        mode = runtime_cfg.get("mode", "test")
        console = runtime_cfg.get("console", True)
        debug = runtime_cfg.get("debug", False)

        # Resolve model name for checkpoint consistency
        if config.model.name == "autoencoder":
            variant = config.model.params.get("variant", "simple")
            if variant not in {"simple", "vae"}:
                raise ValueError(f"Unsupported autoencoder variant: {variant!r}")

            resolved_model_name = (
                f"autoencoder_{variant}"  # ✅ use for checkpoints only
            )
            print(f"[Resolved] Using model checkpoint name: {resolved_model_name}")
        else:
            resolved_model_name = config.model.name

        if logger is None:
            logger = RefraktLogger(
                model_name=resolved_model_name,
                log_dir=log_dir,
                log_types=log_types,
                console=console,
                debug=debug,
            )

        logger.log_config(OmegaConf.to_container(config, resolve=True))

        # === Set up modules and device ===
        modules = {
            "get_model": get_model,
            "get_loss": get_loss,
            "get_trainer": get_trainer,
            "get_wrapper": get_wrapper,
        }
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # === Dataset, Model, Loss ===
        dataloader = _build_test_loader(config)

        model_cls = get_model(config.model.name)
        model = build_model(
            config,
            modules={
                "get_model": get_model,
                "get_wrapper": get_wrapper,
                "model": model_cls,
            },
            device=device,
        )

        loss_fn = build_loss(config, modules=modules, device=device)

        # === Artifact dumper ===
        artifact_log_every = config.get("artifacts", {}).get("log_every", 1)
        artifact_enabled = config.get("artifacts", {}).get("enabled", True)
        artifact_dumper = ArtifactDumper(
            enabled=artifact_enabled,
            base_path="./artifacts",
            model_name=resolved_model_name,
            log_every=artifact_log_every,
            logger=logger,
        )

        # === Trainer ===
        trainer = initialize_trainer(
            cfg=config,
            model=model,
            train_loader=dataloader,
            val_loader=dataloader,
            loss_fn=loss_fn,
            optimizer=None,
            scheduler=None,
            device=device,
            modules=modules,
            save_dir=None,
        )
        trainer.model_name = resolved_model_name

        # === Load Checkpoint ===
        trainer.logger = logger
        trainer.artifact_dumper = artifact_dumper

        if not os.path.exists(model_path):
            base_path = os.path.splitext(model_path)[0]
            candidates = glob.glob(f"{base_path}_*.pth")
            if candidates:
                model_path = max(candidates, key=os.path.getmtime)
                logger.warning(f"Using available checkpoint: {model_path}")
            else:
                raise FileNotFoundError(f"No model found at {model_path}")

        logger.info(f"Loading model from {model_path}")
        trainer.load(path=model_path, suffix="latest")

        # === Evaluation Phase ===
        logger.info("🧪 Running evaluation...")
        eval_results = trainer.evaluate()

        fusion_cfg = config.model.get("fusion")
        if fusion_cfg:
            fusion_type = fusion_cfg.type
            fusion_model_key = fusion_cfg.model
            fusion_model_path = os.path.join(
                config.trainer.params.save_dir, f"{config.model.name}_fusion.joblib"
            )

            if os.path.exists(fusion_model_path):
                logger.info(f"[FUSION] Found fusion head at {fusion_model_path}")

                if fusion_type == "sklearn":
                    fusion_head = SklearnWrapper.load(
                        fusion_model_key, fusion_model_path
                    )
                elif fusion_type == "cuml":
                    fusion_head = CuMLWrapper.load(
                        fusion_model_key, fusion_model_path
                    )
                else:
                    raise ValueError(f"[FUSION] Unsupported fusion type: {fusion_type}")

                fusion_trainer = FusionTrainer(
                    model=model,
                    fusion_head=fusion_head,
                    train_loader=dataloader,  # ✅ we reuse val_loader for feature extraction
                    val_loader=dataloader,  # ✅ same again for accuracy
                    device=device,
                    artifact_dumper=artifact_dumper,
                    model_name=config.model.name,
                )

                fusion_acc = fusion_trainer.evaluate()
                logger.info(
                    f"[FUSION] Validation accuracy (fusion head): {fusion_acc:.4f}"
                )
            else:
                logger.warning(
                    f"[FUSION] No fusion model found at: {fusion_model_path}"
                )

        # === Log model outputs per batch ===
        if hasattr(logger, "wandb") and hasattr(logger.wandb, "step"):
            logger.wandb.step = 0

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if isinstance(batch, torch.Tensor):
                    inputs = batch

                elif isinstance(batch, dict):
                    # ✅ Check common input keys
                    inputs = (
                        batch.get("input")
                        or batch.get("image")
                        or batch.get("lr")  # <- SRGAN uses "lr" as input
                    )
                    if inputs is None:
                        raise ValueError(
                            f"❌ Inference input could not be resolved from batch keys: {list(batch.keys())}"
                        )

                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0]

                else:
                    raise TypeError(f"Unsupported batch type: {type(batch)}")
                inputs = inputs.to(device)
                output = model(inputs)

                if isinstance(output, ModelOutput):
                    artifact_dumper.log_output(output, batch_id=i)
                else:
                    logger.warning(
                        f"Model output at batch {i} is not a ModelOutput instance."
                    )

        artifact_path = f"./artifacts/{config.model.name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_outputs.pt"
        artifact_dumper.save(artifact_path)

        logger.info("✅ Evaluation completed.")
        return {
            "model": trainer.model,
            "evaluation_results": eval_results,
            "config": config,
            "artifacts_path": artifact_path,
        }

    except Exception as e:
        logger = logger or get_global_logger()
        logger.error(f"\n❌ Evaluation failed: {str(e)}")
        logger.error(traceback.format_exc())
        return None
