import numpy as np
import pandas as pd
from monai import data
from monai.utils.type_conversion import convert_data_type
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from src.utils.misc import create_dataset
from src.data.transforms import loading_transforms
from typing import List, Tuple, Dict, Any, Optional


def custom_collate_fn(batch: List[Any]) -> Any:
    """
    Custom collate function to filter out None items from the batch.
    """
    batch = [item for item in batch if item is not None]
    return torch.utils.data.dataloader.default_collate(batch)


class PretrainDataset(Dataset):
    def __init__(self, config: Any, csv_file: str, data_augmentation: Any, cache_dir: Optional[str] = None):
        """
        Dataset class for pre-training.

        Args:
            config: Configuration object.
            csv_file: Path to the CSV file containing image paths.
            data_augmentation: Data augmentation transforms.
            cache_dir: Directory for caching the dataset.
        """
        self.roi = config.MODEL.ROI
        self.in_channels = config.MODEL.IN_CHANS
        self.model_name = config.MODEL.NAME
        self.num_global_crop = config.DINO.GLOBAL_CROP_NUM
        self.num_local_crop = config.DINO.LOCAL_CROP_NUM
        self.global_crop_size = config.DINO.GLOBAL_CROP_SIZE
        self.local_crop_size = config.DINO.LOCAL_CROP_SIZE
        self.data = pd.read_csv(csv_file)
        self.load = loading_transforms(self.roi, self.in_channels)
        self.cache_dir = cache_dir
        self.cache_dataset = data.PersistentDataset(
            data=[{"image": d} for d in self.data['img_path']], 
            transform=self.load, 
            cache_dir=self.cache_dir,
        )
        self.data_augmentation = data_augmentation
        placeholder_image = convert_data_type(torch.zeros(self.in_channels, *self.roi, dtype=torch.float16), output_type=data.MetaTensor)[0]
        if 'dino' in self.model_name:
            self.placeholder_image = [placeholder_image for _ in range(self.num_local_crop + self.num_global_crop)]
        else:
            self.placeholder_image = placeholder_image
        if 'dino' in self.model_name:
            self.placeholder_dict = {
                "image": self.placeholder_image,
                "foreground_start_coord": np.zeros(len(self.roi), dtype=int),
                "foreground_end_coord": np.zeros(len(self.roi), dtype=int),
            }
        else:
            self.placeholder_dict = {
                "image": self.placeholder_image,
                "image_meta_dict": {'filename_or_obj': 'None'},
                "foreground_start_coord": np.zeros(len(self.roi), dtype=int),
                "foreground_end_coord": np.zeros(len(self.roi), dtype=int),
            }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        try:
            image = self.cache_dataset.__getitem__(idx)
            if image['image'].shape[0] != self.in_channels:
                print(f"Wrong number of channels in index {idx}: {image['image'].shape}")
                if self.model_name != 'dino':
                    return self.placeholder_dict['image']
                else:
                    return [torch.randn(self.in_channels, *self.roi, dtype=torch.float16) for _ in range(self.num_local_crop + self.num_global_crop)]
            elif image.keys() != self.placeholder_dict.keys():
                print(f"Wrong keys in index {idx}: {image.keys()}")
                if "dino" not in self.model_name:
                    return self.placeholder_dict['image']
                else:
                    return [torch.randn(self.in_channels, *self.roi, dtype=torch.float16) for _ in range(self.num_local_crop + self.num_global_crop)]
            else:
                if "dino" not in self.model_name:
                    if self.data_augmentation:
                        image = self.data_augmentation(image)
                    return image['image']
                else:
                    if self.data_augmentation:
                        image = self.data_augmentation(image['image'])
                    return image
        except Exception as e:
            print(f"Error loading index {idx}: {e}")
            return self.placeholder_dict['image']


def get_pretrain_dataloaders(config: Any, augs: List[Any]) -> Tuple[data.ThreadDataLoader, data.ThreadDataLoader, data.ThreadDataLoader]:
    """
    Get dataloaders for pre-training.

    Args:
        config: Configuration object.
        augs: List of augmentations for training, validation, and testing.

    Returns:
        Tuple containing train, validation, and test dataloaders.
    """
    # Get augmentations
    imtrans, imvals, imtests = augs[0], augs[1], augs[2]
    # Get data parameters
    batch_size = config.DATA.BATCH_SIZE
    cache_dir = config.DATA.CACHE_DIR
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    # Train dataloader
    train_ds = PretrainDataset(config, 
                               csv_file=config.DATA.TRAIN_CSV_PATH, 
                               data_augmentation=imtrans, 
                               cache_dir=cache_dir
                               )
    sampler_train = data.DistributedSampler(
        train_ds, 
        shuffle=False, 
        num_replicas=num_tasks, 
        rank=global_rank
    )
    train_loader = data.ThreadDataLoader(
        train_ds, 
        batch_size=batch_size, 
        sampler=sampler_train, 
        collate_fn=custom_collate_fn, 
        num_workers=config.DATA.NUM_WORKERS, 
        pin_memory=config.DATA.PIN_MEMORY
    )

    # Validation dataloader
    val_ds = PretrainDataset(
        config, 
        csv_file=config.DATA.VAL_CSV_PATH, 
        data_augmentation=imvals, 
        cache_dir=cache_dir
    )
    sampler_val = data.DistributedSampler(
        val_ds, 
        shuffle=False, 
        num_replicas=num_tasks, 
        rank=global_rank
    )
    val_loader = data.ThreadDataLoader(
        val_ds, 
        batch_size=batch_size, 
        sampler=sampler_val, 
        collate_fn=custom_collate_fn, 
        num_workers=config.DATA.NUM_WORKERS, 
        pin_memory=config.DATA.PIN_MEMORY
    )

    # Test dataloader
    test_ds = PretrainDataset(
        config, 
        csv_file=config.DATA.TEST_CSV_PATH, 
        data_augmentation=imtests, 
        cache_dir=cache_dir
    )
    sampler_test = data.DistributedSampler(
        test_ds, 
        shuffle=False, 
        num_replicas=num_tasks, 
        rank=global_rank
    )
    test_loader = data.ThreadDataLoader(
        test_ds, 
        batch_size=batch_size, 
        sampler=sampler_test, 
        collate_fn=custom_collate_fn, 
        num_workers=config.DATA.NUM_WORKERS, 
        pin_memory=config.DATA.PIN_MEMORY
    )

    return train_loader, val_loader, test_loader


class FinetuneDataset(Dataset):
    def __init__(self, config: Any, files: List[str], label_dict: Dict[str, int], data_augmentation: Any, cache_dir: Optional[str] = None):
        """
        Dataset class for fine-tuning.

        Args:
            config: Configuration object.
            files: List of file paths.
            label_dict: Dictionary mapping file paths to labels.
            data_augmentation: Data augmentation transforms.
            cache_dir: Directory for caching the dataset.
        """
        self.roi = config.MODEL.ROI
        self.in_channels = config.MODEL.IN_CHANS
        self.files = files
        self.cache_dir = cache_dir
        self.label_dict = label_dict
        self.data_augmentation = data_augmentation
        self.load = loading_transforms(self.roi, self.in_channels)
        self.cache_dataset = data.PersistentDataset(data=files, transform=self.load, cache_dir=cache_dir)
        self.placeholder_image = convert_data_type(torch.zeros(self.in_channels, *self.roi, dtype=torch.float16), output_type=data.MetaTensor)[0]
        self.placeholder_dict = {
            "image": self.placeholder_image,
            "image_meta_dict": {'filename_or_obj': 'None'},
            "foreground_start_coord": np.zeros(len(self.roi), dtype=int),
            "foreground_end_coord": np.zeros(len(self.roi), dtype=int),
        }

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[Any, int, str]:
        try:
            item = self.cache_dataset.__getitem__(idx)
            fname = item['image_meta_dict']['filename_or_obj']
            if item['image'].shape[0] != self.in_channels:
                print(f"Wrong number of channels in index {idx}: {item['image'].shape}")
                return self.placeholder_dict['image'], 0, fname
            elif item.keys() != self.placeholder_dict.keys():
                print(f"Wrong keys in index {idx}: {item.keys()}")
                return self.placeholder_dict['image'], 0, fname
            else:
                if self.data_augmentation:
                    item = self.data_augmentation(item)
                return item['image'], self.label_dict[fname], fname
        except Exception as e:
            print(f"Error loading index {idx}: {e}")
            return self.placeholder_dict['image'], 0, None


def get_finetune_dataloaders(config: Any, augs: List[Any]) -> Tuple[data.ThreadDataLoader, data.ThreadDataLoader, data.ThreadDataLoader, Optional[torch.Tensor]]:
    """
    Get dataloaders for fine-tuning.

    Args:
        config: Configuration object.
        augs: List of augmentations for training, validation, and testing.

    Returns:
        Tuple containing train, validation, and test dataloaders, and class weights.
    """
    # Get class mapping
    if config.DATA.DATASET == 'nyu' or config.DATA.DATASET == 'longisland':
        class_mapping = {'cancer': 1, 'hydrocephalus': 2, 'edema': 3, 'dementia': 4, 'IPH': 5, 'IVH': 6, 'SDH': 7, 'EDH': 8, 'SAH': 9, 'ICH': 10, 'fracture': 11}
    elif config.DATA.DATASET == 'rsna':
        class_mapping = {'epidural': 1, 'intraparenchymal': 2, 'intraventricular': 3, 'subarachnoid': 4, 'subdural': 5, 'any': 6}
    elif config.DATA.DATASET == 'cq500':
        class_mapping = {'ICH': 1, 'IPH': 2, 'IVH': 3, 'SDH': 4, 'EDH': 5, 'SAH': 6, 'BleedLocation-Left': 7, 'BleedLocation-Right': 8, 'ChronicBleed': 9, 'Fracture': 10, 'CalvarialFracture': 11, 'OtherFracture': 12, 'MassEffect': 13, 'MidlineShift': 14}
    else:
        raise ValueError(f"Unrecognized dataset: {config.DATA.DATASET}")

    # # Get class weights
    class_idx = class_mapping.get(config.TRAIN.LABEL_NAME, None)
    imtrans, imvals, imtests = augs[0], augs[1], augs[2]
    class_weights = None
    batch_size = config.DATA.BATCH_SIZE
    cache_dir = config.DATA.CACHE_DIR
    num_classes = config.DATA.NUM_CLASSES

    # Load data
    df_train, df_val, df_test = pd.read_csv(config.DATA.TRAIN_CSV_PATH), pd.read_csv(config.DATA.VAL_CSV_PATH), pd.read_csv(config.DATA.TEST_CSV_PATH)
    img_train = list(df_train['img_path'])
    img_val = list(df_val['img_path'])
    img_test = list(df_test['img_path'])
    train_files = create_dataset(img_train, None)
    val_files = create_dataset(img_val, None)
    test_files = create_dataset(img_test, None)

    # Get class weights
    if class_idx is not None:
        label_train = list(df_train.iloc[:, class_idx])
        if num_classes != 1:
            y_train = np.array(label_train)
            class_counts = np.bincount(y_train)
            total_samples = len(y_train)
            class_weights = torch.tensor([1 / (count / total_samples) for count in class_counts], dtype=torch.float)

    train_label_dict = df_train.set_index('img_path').iloc[:, class_idx-1].to_dict()
    val_label_dict = df_val.set_index('img_path').iloc[:, class_idx-1].to_dict()
    test_label_dict = df_test.set_index('img_path').iloc[:, class_idx-1].to_dict()

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    # Train dataloader
    train_ds = FinetuneDataset(
        config, 
        files=train_files, 
        label_dict=train_label_dict, 
        data_augmentation=imtrans, 
        cache_dir=cache_dir
    )
    sample_size = 500
    sample_weights = np.array([class_weights[t] for t in y_train])
    sampler_train = data.DistributedWeightedRandomSampler(
        dataset=train_ds, 
        weights=sample_weights, 
        num_samples_per_rank=sample_size, 
        rank=global_rank
    )
    train_loader = data.ThreadDataLoader(
        train_ds, 
        batch_size=batch_size, 
        sampler=sampler_train, 
        collate_fn=custom_collate_fn, 
        num_workers=config.DATA.NUM_WORKERS, 
        pin_memory=config.DATA.PIN_MEMORY
    )

    # Validation dataloader
    val_ds = FinetuneDataset(
        config, 
        files=val_files, 
        label_dict=val_label_dict, 
        data_augmentation=imvals, 
        cache_dir=cache_dir
    )
    sampler_val = data.DistributedSampler(
        dataset=val_ds, 
        shuffle=False, 
        num_replicas=num_tasks, 
        rank=global_rank
    )
    val_loader = data.ThreadDataLoader(
        val_ds, 
        batch_size=batch_size, 
        sampler=sampler_val, 
        collate_fn=custom_collate_fn,
        num_workers=config.DATA.NUM_WORKERS, 
        pin_memory=config.DATA.PIN_MEMORY
    )

    # Test dataloader
    test_ds = FinetuneDataset(
        config, 
        files=test_files, 
        label_dict=test_label_dict, 
        data_augmentation=imtests, 
        cache_dir=cache_dir
    )
    sampler_test = data.DistributedSampler(
        dataset=test_ds, 
        shuffle=False, 
        num_replicas=num_tasks, 
        rank=global_rank
    )
    test_loader = data.ThreadDataLoader(
        test_ds, 
        batch_size=batch_size, 
        sampler=sampler_test, 
        collate_fn=custom_collate_fn, 
        num_workers=config.DATA.NUM_WORKERS, 
        pin_memory=config.DATA.PIN_MEMORY
    )

    return train_loader, val_loader, test_loader, class_weights


def get_fewshots_dataloaders(config: Any, augs: List[Any]) -> Tuple[data.ThreadDataLoader, data.ThreadDataLoader, data.ThreadDataLoader, Optional[torch.Tensor]]:
    """
    Get dataloaders for few-shot learning.

    Args:
        config: Configuration object.
        augs: List of augmentations for training, validation, and testing.

    Returns:
        Tuple containing train, validation, and test dataloaders, and class weights.
    """
    # Get class mapping
    if config.DATA.DATASET == 'nyu' or config.DATA.DATASET == 'longisland':
        class_mapping = {'cancer': 1, 'hydrocephalus': 2, 'edema': 3, 'dementia': 4, 'IPH': 5, 'IVH': 6, 'SDH': 7, 'EDH': 8, 'SAH': 9, 'ICH': 10, 'fracture': 11}
    elif config.DATA.DATASET == 'rsna':
        class_mapping = {'epidural': 1, 'intraparenchymal': 2, 'intraventricular': 3, 'subarachnoid': 4, 'subdural': 5, 'any': 6}
    elif config.DATA.DATASET == 'cq500':
        class_mapping = {'ICH': 1, 'IPH': 2, 'IVH': 3, 'SDH': 4, 'EDH': 5, 'SAH': 6, 'BleedLocation-Left': 7, 'BleedLocation-Right': 8, 'ChronicBleed': 9, 'Fracture': 10, 'CalvarialFracture': 11, 'OtherFracture': 12, 'MassEffect': 13, 'MidlineShift': 14}
    else:
        raise ValueError(f"Unrecognized dataset: {config.DATA.DATASET}")
    # Get class index
    class_idx = class_mapping.get(config.TRAIN.LABEL_NAME, None)
    imtrans, imvals, imtests = augs[0], augs[1], augs[2]
    class_weights = None
    batch_size = config.DATA.BATCH_SIZE
    cache_dir = config.DATA.CACHE_DIR

    # Load data
    df_train, df_val, df_test = pd.read_csv(config.DATA.TRAIN_CSV_PATH), pd.read_csv(config.DATA.VAL_CSV_PATH), pd.read_csv(config.DATA.TEST_CSV_PATH)
    min_size = config.DATA.FEW_SHOTS
    df_train = df_train.groupby(config.TRAIN.LABEL_NAME).sample(n=min_size, replace=True)
    img_train = list(df_train['img_path'])
    img_val = list(df_val['img_path'])
    img_test = list(df_test['img_path'])
    train_files = create_dataset(img_train, None)
    val_files = create_dataset(img_val, None)
    test_files = create_dataset(img_test, None)

    # Get class weights
    train_label_dict = df_train.set_index('img_path').iloc[:, class_idx-1].to_dict()
    val_label_dict = df_val.set_index('img_path').iloc[:, class_idx-1].to_dict()
    test_label_dict = df_test.set_index('img_path').iloc[:, class_idx-1].to_dict()

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    # Train dataloader
    train_ds = FinetuneDataset(
        config, 
        files=train_files, 
        label_dict=train_label_dict, 
        data_augmentation=imtrans, 
        cache_dir=cache_dir)
    sampler_train = data.DistributedSampler(
        dataset=train_ds, 
        shuffle=True, 
        num_replicas=num_tasks, 
        rank=global_rank)
    train_loader = data.ThreadDataLoader(
        train_ds, 
        batch_size=batch_size, 
        sampler=sampler_train, 
        collate_fn=custom_collate_fn, 
        num_workers=config.DATA.NUM_WORKERS, 
        pin_memory=config.DATA.PIN_MEMORY
    )

    # Validation dataloader
    val_ds = FinetuneDataset(
        config, 
        files=val_files, 
        label_dict=val_label_dict, 
        data_augmentation=imvals, 
        cache_dir=cache_dir
    )
    sampler_val = data.DistributedSampler(
        dataset=val_ds, 
        shuffle=False, 
        num_replicas=num_tasks, 
        rank=global_rank
    )
    val_loader = data.ThreadDataLoader(
        val_ds, 
        batch_size=batch_size, 
        sampler=sampler_val, 
        collate_fn=custom_collate_fn, 
        num_workers=config.DATA.NUM_WORKERS, 
        pin_memory=config.DATA.PIN_MEMORY
    )

    # Test dataloader
    test_ds = FinetuneDataset(
        config, 
        files=test_files, 
        label_dict=test_label_dict, 
        data_augmentation=imtests, 
        cache_dir=cache_dir
    )
    sampler_test = data.DistributedSampler(
        dataset=test_ds, 
        shuffle=False, 
        num_replicas=num_tasks, 
        rank=global_rank
    )
    test_loader = data.ThreadDataLoader(
        test_ds, 
        batch_size=batch_size, 
        sampler=sampler_test, 
        collate_fn=custom_collate_fn, 
        num_workers=config.DATA.NUM_WORKERS, 
        pin_memory=config.DATA.PIN_MEMORY
    )

    return train_loader, val_loader, test_loader, class_weights
