import torch
from torch.nn.utils.rnn import pad_sequence


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
    result_batch["audio_path"] = [item["audio_path"] for item in dataset_items]
    result_batch["audio"] = [item["audio"] for item in dataset_items]

    result_batch["mel_spec"] = []
    result_batch["spec_length"] = torch.tensor(
        [item["mel_spec"].shape[2] for item in dataset_items]
    )
    for item in dataset_items:
        result_batch["mel_spec"].append(item["mel_spec"].squeeze(0).T)

    result_batch["mel_spec"] = pad_sequence(
        result_batch["mel_spec"], batch_first=True
    ).permute(0, 2, 1)

    return result_batch
