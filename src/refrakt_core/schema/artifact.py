import os
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Union, List, Any
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.schema.loss_output import LossOutput


class ArtifactDumper:
    """
    Logs and saves model outputs and losses for visualization, analysis, or explainability.

    Stores per-batch outputs (logits, embeddings, attention maps, reconstructions),
    targets, predictions, filenames, and optional loss info.
    """

    def __init__(
        self,
        enabled: bool = True,
        model_name: Optional[str] = None,
        base_path: str = "./artifacts",
        auto_flush: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.enabled = enabled
        self.model_name = model_name
        self.base_path = base_path
        self.auto_flush = auto_flush
        self.buffer: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = metadata or {}
        os.makedirs(self.base_path, exist_ok=True)

    def log_output(
        self,
        output: ModelOutput,
        batch_id: Union[int, str],
        targets: Optional[torch.Tensor] = None,
        filenames: Optional[List[str]] = None,
    ):
        if not self.enabled:
            return

        record = {}

        if output.logits is not None:
            logits = output.logits.detach().cpu()
            record["logits"] = logits
            record["probs"] = F.softmax(logits, dim=1)
            record["preds"] = torch.argmax(logits, dim=1)

        if output.embeddings is not None:
            record["embeddings"] = output.embeddings.detach().cpu()

        if output.reconstruction is not None:
            record["reconstruction"] = output.reconstruction.detach().cpu()

        if output.attention_maps is not None:
            record["attention_maps"] = output.attention_maps.detach().cpu()

        if output.generated_image is not None:
            record["generated"] = output.generated_image.detach().cpu()

        if output.custom is not None:
            record["custom"] = output.custom.detach().cpu()

        if targets is not None:
            record["targets"] = targets.detach().cpu()

        if filenames is not None:
            record["filenames"] = filenames

        self.buffer[str(batch_id)] = record

        if self.auto_flush:
            self.save(filename=f"batch_{batch_id}.pt")

    def log_loss(self, loss: Union[LossOutput, Dict[str, torch.Tensor]], batch_id: Union[int, str]):
        if not self.enabled:
            return

        record = self.buffer.get(str(batch_id), {})
        if isinstance(loss, LossOutput):
            record["loss_total"] = float(loss.total)
            record["loss_components"] = {k: float(v.item()) for k, v in loss.components.items()}
        elif isinstance(loss, dict):
            record["loss_dict"] = {k: float(v.item()) for k, v in loss.items()}

        self.buffer[str(batch_id)] = record

    def save(self, filename: Optional[str] = None):
        if not self.enabled:
            return

        if filename is None:
            filename = f"artifacts_{self.model_name or 'model'}.pt"

        save_data = {
            "metadata": self.metadata,
            "outputs": self.buffer,
        }

        full_path = os.path.join(self.base_path, filename)
        torch.save(save_data, full_path)


    def reset(self):
        self.buffer = {}

    def get_batch(self, batch_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        return self.buffer.get(str(batch_id))

    def summary(self) -> Dict[str, Any]:
        num_batches = len(self.buffer)
        num_preds = sum(batch.get("logits", torch.empty(0)).shape[0] for batch in self.buffer.values())
        return {
            "total_batches": num_batches,
            "total_predictions": num_preds,
            "metadata_keys": list(self.metadata.keys()),
        }
