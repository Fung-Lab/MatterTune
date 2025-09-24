from __future__ import annotations

import logging
import os
import shutil
import tempfile
from typing_extensions import override

import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import BasePredictionWriter

logger = logging.getLogger(__name__)

class CustomWriter(BasePredictionWriter):
    """
    Pytorch Lightning Callback, for saving predictions from multiple GPUs during prediction.
    Requirements:
    1. collect all predictions from different GPUs
    2. follow the input order of the dataloader
    """
    def __init__(self, write_interval="epoch") -> None:
        super().__init__(write_interval)
        self.output_dir = None

    @override
    def write_on_epoch_end(self, trainer, pl_module, predictions:list[dict[str, torch.Tensor]], batch_indices):
        """
        Called at the end of each epoch, saves the predictions and corresponding batch indices
        """
        # Create a temporary directory to store predictions only on global_rank 0
        if trainer.is_global_zero:
            output_dir = [tempfile.mkdtemp()]
            logger.info("Created temporary folder to store predictions: {}.".format(output_dir[0]))
            os.makedirs(output_dir[0], exist_ok=True)
        else:
            output_dir = [None]
        # Broadcast the output_dir to all processes
        dist.broadcast_object_list(output_dir)
        # make sure all processes have the same output_dir
        dist.barrier()
        self.output_dir = output_dir[0]
        assert os.path.exists(self.output_dir), f"Output directory {self.output_dir} does not exist."
        
        flat_predictions = [p for batch in predictions for p in batch]
        flat_indices = [i for batch in batch_indices[0] for i in batch]

        torch.save(flat_predictions, os.path.join(self.output_dir, f"pred_{trainer.global_rank}.pt")) # type: ignore
        torch.save(flat_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt")) # type: ignore

    def gather_all_predictions(self):
        """
        Load all saved predictions and corresponding indices from the temporary folder,
        sort them according to the batch indices, and aggregate the predictions into a single dictionary.
        """
        files = sorted(os.listdir(self.output_dir))
        pred_files = sorted([f for f in files if f.startswith("pred_")])
        idx_files = sorted([f for f in files if f.startswith("batch_indices_")])

        all_items = []

        for pred_file, idx_file in zip(pred_files, idx_files):
            global_rank_pred = int(pred_file.split("_")[-1].split(".")[0])
            global_rank_idx = int(idx_file.split("_")[-1].split(".")[0])
            assert global_rank_pred == global_rank_idx, f"Mismatch in global rank: {global_rank_pred} vs {global_rank_idx}"
            pred_list_of_dict = torch.load(os.path.join(self.output_dir, pred_file)) # type: ignore
            indices = torch.load(os.path.join(self.output_dir, idx_file)) # type: ignore
            assert len(pred_list_of_dict) == len(indices), f"Mismatch in length: {len(pred_list_of_dict)} vs {len(indices)}"
            if isinstance(indices, torch.Tensor):
                indices = indices.tolist()

            batch_size = len(pred_list_of_dict)

            for i in range(batch_size): # type: ignore
                sample_pred = pred_list_of_dict[i]
                all_items.append((indices[i], sample_pred))

        all_items = sorted(all_items, key=lambda x: x[0])
        final_predictions = []
        if all_items:
            for _, sample_pred in all_items:
                final_predictions.append(sample_pred)
        return final_predictions

    def cleanup(self):
        """
        Cleanup the temporary folder used for storing predictions.
        """
        logger.info("Cleanup temporary folder: {}.".format(self.output_dir))
        shutil.rmtree(self.output_dir) # type: ignore
