import polars as pl
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from monai import transforms
from typing import List, Tuple, Optional
from src.fm_ct.data.transforms import MultipleWindowScaleStack
from transformers import AutoTokenizer


class HeadCTDataset(Dataset):
    """
    Dataset class for Head CT objectives using pre-trained ViT models.

    This dataset handles:
    - Loading NIfTI files
    - Applying multiple window scaling (bone, soft tissue, brain windows)
    - Preprocessing for ViT model input
    - Batch processing for feature extraction
    """

    def __init__(
        self,
        image_file_location: str,
        objective_column: str,
        use_cached_images: bool = True,
        tokenizer_model_name: Optional[str] = None,
        max_text_length: int = 512,
        roi: Tuple[int, int, int] = (96, 96, 96),
        window_sizes: List[Tuple[int, int]] = [(40, 80), (80, 200), (600, 2800)],
        pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        axcodes: str = "RAS",
        mode: int = 3,
        allow_missing_keys: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            image_file_location: Path to parquet file with image paths and conditions
            tokenizer_model_name: Name of the HuggingFace model to use for tokenization
            max_text_length: Maximum length for tokenized text
            roi: Region of interest size (D, H, W)
            window_sizes: List of (center, width) tuples for windowing
            pixdim: Target pixel dimensions for resampling
            axcodes: Target orientation code
            mode: Interpolation mode for resampling
            allow_missing_keys: Whether to allow missing keys in transforms
        """
        self.image_df: pl.DataFrame = pl.read_parquet(image_file_location)

        image_path_col: str = "img_path" if not use_cached_images else "cached_path"
        self.image_paths: List[str] = self.image_df[image_path_col].to_list()
        self.objective_text: List[str] = self.image_df[objective_column].to_list()

        self.roi: Tuple[int, int, int] = roi
        self.window_sizes: List[Tuple[int, int]] = window_sizes

        # Initialize tokenizer if model name is provided
        self.tokenizer = None
        self.max_text_length = max_text_length
        if tokenizer_model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)

        # Create the data list in MONAI format
        self.data: List[dict] = [
            {"image_item": {"image": path}, "objective": obj}
            for path, obj in zip(self.image_paths, self.objective_text)
        ]

        # Initialize transforms - only needed if not using cached images (preferred)
        self.transforms = None
        if not use_cached_images:
            self.transforms = self._build_transforms(
                roi, window_sizes, pixdim, axcodes, mode, allow_missing_keys
            )

    def _build_transforms(
        self,
        roi: Tuple[int, int, int],
        window_sizes: List[Tuple[int, int]],
        pixdim: Tuple[float, float, float],
        axcodes: str,
        mode: int,
        allow_missing_keys: bool,
    ) -> transforms.Compose:
        """Build the preprocessing transform pipeline."""

        windowing_tran = MultipleWindowScaleStack(
            keys=["image"],
            window_sizes=window_sizes,
        )

        transform_list = [
            transforms.LoadImaged(
                keys=["image"],
                image_only=True,
                allow_missing_keys=allow_missing_keys,
            ),
            transforms.EnsureChannelFirstd(
                keys=["image"],
                allow_missing_keys=allow_missing_keys,
            ),
            transforms.Orientationd(
                keys=["image"],
                axcodes=axcodes,
                allow_missing_keys=allow_missing_keys,
            ),
            transforms.Spacingd(
                keys=["image"],
                pixdim=pixdim,
                mode=mode,
                allow_missing_keys=allow_missing_keys,
            ),
            transforms.CropForegroundd(
                keys=["image"],
                source_key="image",
                allow_smaller=False,
                allow_missing_keys=allow_missing_keys,
            ),
            transforms.Resized(
                keys=["image"],
                spatial_size=roi,
                allow_missing_keys=allow_missing_keys,
            ),
            windowing_tran,
            transforms.ToTensord(
                keys=["image"],
                allow_missing_keys=allow_missing_keys,
            ),
        ]

        return transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample_data = self.data[idx].copy()

        if self.transforms:
            sample_image = sample_data["image_item"]
            transformed_sample = self.transforms(sample_image)
            image_tensor = transformed_sample["image"]
        else:
            cached_path = sample_data["image_item"]["image"]
            image_array = np.load(cached_path)
            image_tensor = torch.from_numpy(image_array)

        result = {
            "image": image_tensor,
            "objective": sample_data["objective"],
        }

        if self.tokenizer:
            tokenized = self.tokenizer(
                sample_data["objective"],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_text_length,
            )
            result["input_ids"] = tokenized["input_ids"].squeeze(0)
            result["attention_mask"] = tokenized["attention_mask"].squeeze(0)

        return result


def collate_fn_dynamic_padding(batch):
    """
    Custom collate function that pads sequences dynamically to the longest in the batch.
    """
    if len(batch) > 0:
        expected_shape = batch[0]["image"].shape
        batch = [item for item in batch if item["image"].shape == expected_shape]

    if len(batch) == 0:
        return None

    images = torch.stack([item["image"] for item in batch])
    objectives = [item["objective"] for item in batch]

    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]

    max_len = max(len(ids) for ids in input_ids)

    padded_input_ids = []
    padded_attention_masks = []

    for ids, mask in zip(input_ids, attention_masks):
        padding_length = max_len - len(ids)
        padded_input_ids.append(
            torch.cat([ids, torch.zeros(padding_length, dtype=ids.dtype)])
        )
        padded_attention_masks.append(
            torch.cat([mask, torch.zeros(padding_length, dtype=mask.dtype)])
        )

    return {
        "image": images,
        "objective": objectives,
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_masks),
    }


def create_head_ct_dataloader(
    image_file_location: str,
    objective_column: str,
    batch_size: int = 4,
    num_workers: int = 1,
    pin_memory: bool = True,
    shuffle: bool = False,
    use_cached_images: bool = True,
    rank: int = 0,
    world_size: int = 1,
    tokenizer_model_name: Optional[str] = None,
    max_text_length: int = 512,
    roi: Tuple[int, int, int] = (96, 96, 96),
    window_sizes: List[Tuple[int, int]] = [(40, 80), (80, 200), (600, 2800)],
    persistent_workers: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    """
    Create a DataLoader for Head CT feature extraction.

    Args:
        image_file_location: Path to parquet file with image paths
        objective_column: Column name for the objective text
        batch_size: Batch size for the DataLoader
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        shuffle: Whether to shuffle the data
        tokenizer_model_name: Name of HuggingFace model for tokenization
        max_text_length: Maximum length for tokenized text
        roi: Region of interest size
        window_sizes: List of windowing parameters
        **dataset_kwargs: Additional arguments for the dataset

    Returns:
        DataLoader configured for feature extraction
    """
    dataset = HeadCTDataset(
        image_file_location=image_file_location,
        objective_column=objective_column,
        use_cached_images=use_cached_images,
        tokenizer_model_name=tokenizer_model_name,
        max_text_length=max_text_length,
        roi=roi,
        window_sizes=window_sizes,
        **dataset_kwargs,
    )

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        shuffle = False

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn_dynamic_padding,
    )

    return dataloader
