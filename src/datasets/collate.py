import torch
import torch.nn.functional as F

from src.utils.spec_utils import MelSpectrogram, MelSpectrogramConfig


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    result_batch = {}
    mel_spec_transform = MelSpectrogram(MelSpectrogramConfig)
    result_batch["audio"] = torch.stack([item["audio"] for item in dataset_items])
    result_batch["mel_spec"] = mel_spec_transform(result_batch["audio"]).squeeze(1)
    return result_batch
