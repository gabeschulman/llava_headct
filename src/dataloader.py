from functools import partial
import polars as pl
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from monai import transforms
from typing import List, Tuple, Optional
from src.fm_ct.data.transforms import MultipleWindowScaleStack
from transformers import AutoTokenizer
from src.constants import (
    PROMPT_TEMPLATES,
    OBJECTIVE_DICT,
    INDIVIDUAL_CONDITIONS_LIST,
    ABBREVIATED_CONDITIONS_DICT,
)


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
        use_cached_images: bool = True,
        tokenizer_model_name: Optional[str] = None,
        max_text_length: int = 512,
        roi: Tuple[int, int, int] = (96, 96, 96),
        window_sizes: List[Tuple[int, int]] = [(40, 80), (80, 200), (600, 2800)],
        pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        axcodes: str = "RAS",
        mode: int = 3,
        allow_missing_keys: bool = True,
        numpy_seed: int = 1,
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
        np.random.seed(numpy_seed)
        self.image_df: pl.DataFrame = pl.read_parquet(image_file_location).sample(
            fraction=1.0, shuffle=True, seed=42
        )

        self.image_path_col: str = (
            "img_path" if not use_cached_images else "cached_path"
        )

        self.roi: Tuple[int, int, int] = roi
        self.window_sizes: List[Tuple[int, int]] = window_sizes

        # Initialize tokenizer if model name is provided
        self.tokenizer = None
        self.max_text_length = max_text_length
        if tokenizer_model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.sample_weights = []

        # Initialize transforms - only needed if not using cached images (preferred)
        self.transforms = None
        if not use_cached_images:
            self.transforms = self._build_transforms(
                roi, window_sizes, pixdim, axcodes, mode, allow_missing_keys
            )

    @property
    def objectives(self) -> List[str]:
        """Get the list of objectives."""
        return list(OBJECTIVE_DICT.keys())

    @property
    def objective_probabilities(self) -> List[float]:
        """Get the objective probabilities dictionary."""
        return list(OBJECTIVE_DICT.values())

    @property
    def number_of_objectives(self) -> int:
        """Get the number of objectives."""
        return len(OBJECTIVE_DICT)

    @property
    def data(self) -> dict:
        objectives = []
        for row in self.image_df.iter_rows(named=True):
            choice = np.random.choice(self.objectives, p=self.objective_probabilities)
            prompt_text, objective_text = self.map_choice_to_objective(row, choice)
            if objective_text is None or (
                isinstance(objective_text, str) and len(objective_text.strip()) == 0
            ):
                continue
            accession_number = row["accession_num"]
            objectives.append(
                {
                    "image_item": {"image": row[self.image_path_col]},
                    "prompt": prompt_text,
                    "objective_type": choice,
                    "objective": objective_text,
                    "accession_number": accession_number,
                    "narrative": row["findings"] if row["findings"] else "",
                    "impression": row["impression_deid_clean"]
                    if row["impression_deid_clean"]
                    else "",
                    "conditions": "CONDITIONS: " + row["conditions"]
                    if row["conditions"]
                    else "",
                    "has_abnormality": row["has_abnormality"],
                }
            )
            self.sample_weights.append(row["sample_weight"])
        return objectives

    def map_choice_to_objective(self, row: dict, choice: str) -> tuple:
        if choice == "condition_classification":
            return (
                PROMPT_TEMPLATES["condition_classification"],
                "CONDITIONS: " + row["conditions"],
            )
        elif choice == "impression_generation":
            return (
                PROMPT_TEMPLATES["impression_generation"],
                row["impression_deid_clean"],
            )
        elif choice == "narrative_generation":
            return PROMPT_TEMPLATES["narrative_generation"], row["findings"]
        elif choice == "individual_condition_classification":
            condition = np.random.choice(INDIVIDUAL_CONDITIONS_LIST)
            return (
                PROMPT_TEMPLATES["individual_condition_classification"].format(
                    condition=ABBREVIATED_CONDITIONS_DICT.get(condition, condition)
                ),
                "Yes" if row[condition] == 1 else "No",
            )
        else:
            raise ValueError(f"Unsupported objective choice: {choice}")

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

        prompt_text = sample_data["prompt"]
        objective_type = sample_data["objective_type"]
        objective_text = sample_data["objective"]

        if objective_text is None or (
            isinstance(objective_text, str) and len(objective_text.strip()) == 0
        ):
            objective_text = ""

        if objective_type == "individual_condition_classification":
            contrastive_text = sample_data["conditions"]
        else:
            contrastive_text = objective_text

        result = {
            "image": image_tensor,
            "prompt": prompt_text,
            "objective_type": objective_type,
            "objective": objective_text,
            "contrastive_text": contrastive_text,
            "accession_number": sample_data["accession_number"],
            "impression": sample_data["impression"],
            "narrative": sample_data["narrative"],
            "conditions": sample_data["conditions"],
            "has_abnormality": sample_data["has_abnormality"],
        }

        return result


def collate_fn_dynamic_padding(batch, padding_token_id: int, tokenizer):
    """
    Custom collate function that pads sequences dynamically to the longest in the batch.
    Filters out images with incorrect shapes and duplicate accessions.
    Handles all tokenization in main process.
    """
    expected_shape = (3, 96, 96, 96)
    seen_accession_numbers = set()

    batch_clean = []
    for item in batch:
        accession_number = item["accession_number"]
        if (
            accession_number in seen_accession_numbers
            or item["image"].shape != expected_shape
        ):
            continue
        seen_accession_numbers.add(accession_number)
        batch_clean.append(item)

    if len(batch_clean) < 2:
        return None

    batch = batch_clean

    images = torch.stack([item["image"] for item in batch])

    objectives = [item["objective"] for item in batch]
    prompts = [item["prompt"] for item in batch]
    contrastive_texts = [item["contrastive_text"] for item in batch]

    tokenized_objectives = []
    objective_attention_masks = []

    for obj_text in objectives:
        tokens = tokenizer(
            obj_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
        )
        input_ids = torch.cat(
            [tokens["input_ids"].squeeze(0), torch.tensor([tokenizer.eos_token_id])]
        )
        attention_mask = torch.cat(
            [tokens["attention_mask"].squeeze(0), torch.tensor([1])]
        )
        tokenized_objectives.append(input_ids)
        objective_attention_masks.append(attention_mask)

    tokenized_prompts = []
    prompt_attention_masks = []

    for prompt_text in prompts:
        tokens = tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        tokenized_prompts.append(tokens["input_ids"].squeeze(0))
        prompt_attention_masks.append(tokens["attention_mask"].squeeze(0))

    contrastive_input_ids = []
    for cont_text in contrastive_texts:
        tokens = tokenizer(
            cont_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        contrastive_input_ids.append(tokens["input_ids"].squeeze(0))

    max_obj_len = max(len(ids) for ids in tokenized_objectives)
    padded_input_ids = []
    padded_attention_masks = []

    for ids, mask in zip(tokenized_objectives, objective_attention_masks):
        padding_length = max_obj_len - len(ids)
        padded_input_ids.append(
            torch.cat([ids, torch.tensor([padding_token_id]).repeat(padding_length)])
        )
        padded_attention_masks.append(
            torch.cat([mask, torch.zeros(padding_length, dtype=mask.dtype)])
        )

    max_prompt_len = max(len(ids) for ids in tokenized_prompts)
    padded_prompt_input_ids = []
    padded_prompt_attention_masks = []

    for ids, mask in zip(tokenized_prompts, prompt_attention_masks):
        padding_length = max_prompt_len - len(ids)
        padded_prompt_input_ids.append(
            torch.cat([ids, torch.tensor([padding_token_id]).repeat(padding_length)])
        )
        padded_prompt_attention_masks.append(
            torch.cat([mask, torch.zeros(padding_length, dtype=mask.dtype)])
        )

    objective_types = [item["objective_type"] for item in batch]
    accession_numbers = [item["accession_number"] for item in batch]
    narratives = [item["narrative"] for item in batch]
    impressions = [item["impression"] for item in batch]
    conditions = [item["conditions"] for item in batch]
    has_abnormalities = [item["has_abnormality"] for item in batch]

    return {
        "image": images,
        "objective": objectives,
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_masks),
        "prompt_input_ids": torch.stack(padded_prompt_input_ids),
        "prompt_attention_mask": torch.stack(padded_prompt_attention_masks),
        "contrastive_input_ids": contrastive_input_ids,  # (variable length)
        "accession_numbers": accession_numbers,
        "narrative": narratives,
        "impression": impressions,
        "conditions": conditions,
        "objective_type": objective_types,
        "has_abnormality": has_abnormalities,
    }


def create_head_ct_dataloader(
    image_file_location: str,
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
        use_cached_images=use_cached_images,
        tokenizer_model_name=tokenizer_model_name,
        max_text_length=max_text_length,
        roi=roi,
        window_sizes=window_sizes,
        **dataset_kwargs,
    )

    sampler = WeightedRandomSampler(
        weights=dataset.sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )
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
        collate_fn=partial(
            collate_fn_dynamic_padding,
            padding_token_id=dataset.tokenizer.pad_token_id,
            tokenizer=dataset.tokenizer,
        )
        if dataset.tokenizer
        else None,
    )

    return dataloader
