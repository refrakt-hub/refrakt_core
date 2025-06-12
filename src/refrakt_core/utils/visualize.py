import io
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter

def visualize_latent_space_tensorboard(model, dataloader, device, writer: SummaryWriter, step=0, logger=None):
    try:
        model.eval()
        all_latents = []
        all_labels = []

        for batch in dataloader:
            inputs = batch[0].to(device)
            labels = batch[1] if len(batch) > 1 else None

            with torch.no_grad():
                latents = model.get_latent(inputs)
                all_latents.append(latents.cpu())
                if labels is not None:
                    all_labels.append(labels)

        latents = torch.cat(all_latents, dim=0).numpy()
        if all_labels:
            labels = torch.cat(all_labels, dim=0).numpy()
        else:
            labels = None

        # Dimensionality reduction
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        latents_2d = reducer.fit_transform(latents)

        # Create plot
        fig, ax = plt.subplots()
        if labels is not None:
            scatter = ax.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap="tab10", alpha=0.7)
            plt.colorbar(scatter)
        else:
            ax.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.7)

        ax.set_title("Latent Space Projection (PCA)")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")

        # Log to TensorBoard
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        import PIL.Image
        image = PIL.Image.open(buf)
        writer.add_image("latent_projection", torch.tensor(np.array(image)).permute(2, 0, 1), global_step=step)
        buf.close()
        plt.close()
        
        if logger: 
            logger.info(f"Latent space visualization logged to TensorBoard with {latents.shape[0]} points")
    except Exception as e:
        if logger:
            logger.error(f"Error visualizing latent space: {e}")
        else:
            print(f"Error visualizing latent space: {e}")

def visualize_latent_space_wandb(model, dataloader, device, step=0, logger=None):
    try:
        model.eval()
        all_latents, all_labels = [], []

        for batch in dataloader:
            inputs = batch[0].to(device)
            labels = batch[1] if len(batch) > 1 else None

            with torch.no_grad():
                latents = model.get_latent(inputs)
                all_latents.append(latents.cpu())
                if labels is not None:
                    all_labels.append(labels)

        latents = torch.cat(all_latents, dim=0).numpy()
        if all_labels:
            labels = torch.cat(all_labels, dim=0).numpy()
        else:
            labels = None

        from sklearn.decomposition import PCA
        latents_2d = PCA(n_components=2).fit_transform(latents)

        plt.figure(figsize=(8, 6))
        if labels is not None:
            plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap="tab10", alpha=0.7)
            plt.colorbar()
        else:
            plt.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.7)
        plt.title("Latent Space Projection")
        plt.xlabel("z1")
        plt.ylabel("z2")

        # Log to W&B
        wandb.log({"latent_space": wandb.Image(plt)}, step=step)
        plt.close()
        
        if logger:
            logger.info(f"Latent space visualization logged to W&B with {latents.shape[0]} points")
    except Exception as e:
        if logger:
            logger.error(f"Error visualizing latent space: {e}")
        else:
            print(f"Error visualizing latent space: {e}")
