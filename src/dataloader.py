import polars as pl
from torch.utils.data import Dataset, DataLoader
from monai import transforms
from typing import List, Tuple
from src.fm_ct.data.transforms import MultipleWindowScaleStack


class ConditionClassificationDataset(Dataset):
    """
    Dataset class for Head CT condition classification using pre-trained ViT models.

    This dataset handles:
    - Loading NIfTI files
    - Applying multiple window scaling (bone, soft tissue, brain windows)
    - Preprocessing for ViT model input
    - Batch processing for feature extraction
    """

    def __init__(
        self,
        image_file_location: str,
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
            image_paths: List of paths to NIfTI files
            roi: Region of interest size (D, H, W)
            window_sizes: List of (center, width) tuples for windowing
            pixdim: Target pixel dimensions for resampling
            axcodes: Target orientation code
            mode: Interpolation mode for resampling
            allow_missing_keys: Whether to allow missing keys in transforms
        """
        self.image_df: pl.DataFrame = pl.read_parquet(image_file_location)
        self.image_paths: List[str] = self.image_df["img_path"].to_list()
        self.conditions: List[str] = self.image_df["conditions"].to_list()

        self.roi: Tuple[int, int, int] = roi
        self.window_sizes: List[Tuple[int, int]] = window_sizes

        # Create the data list in MONAI format
        self.data: List[dict] = [
            {"image_item": {"image": path}, "condition": cond}
            for path, cond in zip(self.image_paths, self.conditions)
        ]

        # Initialize transforms
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
        sample_image = sample_data["image_item"]
        transformed_sample = self.transforms(sample_image)

        return {
            "image": transformed_sample["image"],
            "condition": sample_data["condition"],
        }


def create_condition_classification_dataloader(
    image_file_location: str,
    batch_size: int = 4,
    num_workers: int = 1,
    pin_memory: bool = True,
    shuffle: bool = False,
    roi: Tuple[int, int, int] = (96, 96, 96),
    window_sizes: List[Tuple[int, int]] = [(40, 80), (80, 200), (600, 2800)],
    **dataset_kwargs,
) -> DataLoader:
    """
    Create a DataLoader for Head CT feature extraction.

    Args:
        image_paths: List of paths to NIfTI files
        batch_size: Batch size for the DataLoader
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        shuffle: Whether to shuffle the data
        roi: Region of interest size
        window_sizes: List of windowing parameters
        **dataset_kwargs: Additional arguments for the dataset

    Returns:
        DataLoader configured for feature extraction
    """
    dataset = ConditionClassificationDataset(
        image_file_location=image_file_location,
        roi=roi,
        window_sizes=window_sizes,
        **dataset_kwargs,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        collate_fn=None,
    )

    return dataloader
