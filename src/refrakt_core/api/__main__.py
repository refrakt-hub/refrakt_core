import argparse
import os
import sys


def main():
    print("==> Refrakt CLI launched")

    parser = argparse.ArgumentParser(description="Refrakt Core Pipeline")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument(
        "--mode",
        choices=["train", "test", "inference", "pipeline"],
        required=True,
        help="Pipeline mode",
    )
    parser.add_argument("--log_dir", default="./logs", help="Log directory path")
    parser.add_argument(
        "--log_type",
        nargs="*",
        choices=["tensorboard", "wandb"],
        default=[],
        help="Logging integrations",
    )
    parser.add_argument("--console", action="store_true", help="Enable console logging")
    parser.add_argument(
        "--model_path", help="Path to model checkpoint (optional for pipeline)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (e.g., registry prints)",
    )
    args = parser.parse_args()

    # Delay ALL imports until after logger configuration
    from omegaconf import OmegaConf

    from refrakt_core.api.core.logger import RefraktLogger
    from refrakt_core.logging import set_global_logger

    cfg = OmegaConf.load(args.config)
    model_name = cfg.model.name

    logger = RefraktLogger(
        model_name=model_name,
        log_dir=args.log_dir,
        log_types=args.log_type,
        console=args.console,
        debug=args.debug,
    )

    logger.info(f"Logging initialized. Log file: {logger.log_file}")
    set_global_logger(logger)

    # Now import pipeline components
    from refrakt_core.api.inference import inference
    from refrakt_core.api.test import test
    from refrakt_core.api.train import train

    try:
        if args.mode == "train":
            logger.info(f"Starting training with config: {args.config}")
            train(args.config, model_path=args.model_path, logger=logger)

        elif args.mode == "test":
            logger.info(f"Starting testing with config: {args.config}")
            test(args.config, model_path=args.model_path, logger=logger)

        elif args.mode == "inference":
            if not args.model_path:
                raise ValueError("--model_path is required for inference mode")
            logger.info(f"Starting inference with config: {args.config}")
            inference(args.config, model_path=args.model_path, logger=logger)

        elif args.mode == "pipeline":
            logger.info("🔁 Starting full pipeline (train → test → inference)")
            cfg = OmegaConf.load(args.config)
            save_dir = cfg.trainer.params.save_dir
            model_name = cfg.trainer.params.model_name
            model_path = os.path.join(save_dir, f"{model_name}.pth")

            logger.info("🚀 Training phase started")
            train(args.config, logger=logger)

            logger.info("🧪 Testing phase started")
            test(args.config, model_path=model_path, logger=logger)

            logger.info("🔮 Inference phase started")
            inference(args.config, model_path=model_path, logger=logger)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    finally:
        logger.info("Finalizing and saving logs...")
        logger.close()


if __name__ == "__main__":
    main()
