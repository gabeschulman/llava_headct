import torch
import numpy as np
from typing import List, Tuple, Dict, Any

from monai import transforms


class MultipleWindowScaleStack(transforms.MapTransform):
    """
    Apply multiple window scaling to the input image and stack the results.
    """

    def __init__(self, keys: List[str], window_sizes: List[Tuple[int, int]]) -> None:
        """
        Args:
            keys (List[str]): Keys of the corresponding items to be transformed.
            window_sizes (List[Tuple[int, int]]): List of window sizes (center, width).
        """
        super().__init__(keys)
        self.keys = keys
        self.window_sizes = window_sizes
        self.scale_transforms = [
            transforms.ScaleIntensityRange(
                a_min=l - w // 2,
                a_max=l + w // 2,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ) for l, w in window_sizes
        ]

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        image = torch.cat([t(d["image"]) for t in self.scale_transforms], dim=0)
        d['image'] = np.array(image)
        return d


class DataAugmentationDINO3D(object):
    """
    Data augmentation for DINO3D.
    """

    def __init__(
        self, 
        final_size: Tuple[int, int, int], 
        global_crops_size: int, 
        local_crops_size: int, 
        local_crops_number: int
    ) -> None:
        """
        Args:
            final_size (Tuple[int, int, int]): Final size of the crops.
            global_crops_size (int): Size of the global crops.
            local_crops_size (int): Size of the local crops.
            local_crops_number (int): Number of local crops.
        """
        flip_and_noise = transforms.Compose([
            transforms.RandFlip(prob=0.2, spatial_axis=0),
            transforms.RandFlip(prob=0.2, spatial_axis=1),
            transforms.RandFlip(prob=0.2, spatial_axis=2),
            transforms.RandShiftIntensity(offsets=0.2, prob=0.5)
        ])

        normalize = transforms.ToTensor()

        globle_min_size = global_crops_size
        local_min_size = local_crops_size
        local_max_size = global_crops_size

        self.global_transfo1 = transforms.Compose([
            transforms.CastToType(dtype=np.float32),
            transforms.ResizeWithPadOrCrop(spatial_size=(224, 224, 224)),
            transforms.RandSpatialCrop(roi_size=globle_min_size, random_center=True, random_size=True),
            transforms.Resize(spatial_size=final_size),
            flip_and_noise,
            transforms.RandGaussianSmooth(sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0), prob=0.2),
            normalize,
        ])
        self.global_transfo2 = transforms.Compose([
            transforms.CastToType(dtype=np.float32),
            transforms.ResizeWithPadOrCrop(spatial_size=(224, 224, 224)),
            transforms.RandSpatialCrop(roi_size=globle_min_size, random_center=True, random_size=True),
            transforms.Resize(spatial_size=final_size),
            flip_and_noise,
            transforms.RandAdjustContrast(gamma=(0.2, 1.0), prob=0.2),
            normalize,
        ])

        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.CastToType(dtype=np.float32),
            transforms.ResizeWithPadOrCrop(spatial_size=(224, 224, 224)),
            transforms.CenterSpatialCrop((192, 192, 192)),
            transforms.RandSpatialCrop(roi_size=local_min_size, max_roi_size=local_max_size, \
                random_center=True, random_size=True),
            transforms.Resize(spatial_size=final_size),
            normalize,
        ])

    def __call__(self, image: np.ndarray) -> List[np.ndarray]:
        crops = [self.global_transfo1(image), self.global_transfo2(image)]
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


def loading_transforms(roi: Tuple[int, int, int], in_channels: int) -> transforms.Compose:
    """
    Define loading transforms based on the number of input channels.

    Args:
        roi (Tuple[int, int, int]): Region of interest size.
        in_channels (int): Number of input channels.

    Returns:
        transforms.Compose: Composed transforms.
    """
    if in_channels == 1:
        windowing_tran = transforms.ScaleIntensityRanged(
            keys=["image"],
            a_min=40 - 150,
            a_max=40 + 150,
            b_min=0.0,
            b_max=1.0,
            clip=True,
            allow_missing_keys=True,
        )
    elif in_channels == 3:
        window_sizes = [(40, 80), (80, 200), (600, 2800)]
        windowing_tran = MultipleWindowScaleStack(
            keys=["image"],
            window_sizes=window_sizes,
        )
    else:
        raise NotImplementedError(f"Channel size {in_channels} is not implemented.")

    trans = transforms.Compose([
        transforms.LoadImaged(
            keys=["image"],
            image_only=False,
            allow_missing_keys=True,
        ),
        transforms.EnsureChannelFirstd(
            keys=["image"],
            allow_missing_keys=True,
        ),
        transforms.Orientationd(
            keys=["image"],
            axcodes="RAS",
            allow_missing_keys=True,
        ),
        transforms.Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode=3,
            allow_missing_keys=True
        ),
        transforms.CropForegroundd(
            keys=["image"],
            source_key="image",
            allow_smaller=False,
            allow_missing_keys=True,
        ),
        windowing_tran,
        transforms.Resized(
            keys=["image"],
            spatial_size=(roi[0], roi[1], roi[2]),
            allow_missing_keys=True,
        ),
        transforms.CastToTyped(
            keys=["image"],
            dtype=np.float16,
            allow_missing_keys=True,
        ),
    ])

    return trans


def mae3d_transforms(config: Any, mode: str = 'train', reshape: bool = False) -> transforms.Compose:
    """
    Define MAE3D transforms based on the mode and reshape flag.

    Args:
        config (Any): Configuration object.
        mode (str): Mode of operation ('train', 'val', 'test').
        reshape (bool): Whether to reshape the image.

    Returns:
        transforms.Compose: Composed transforms.
    """
    if mode in ['train', 'val']:
        trans = transforms.Compose([
            transforms.CastToTyped(
                keys=["image"],
                dtype=np.float32,
                allow_missing_keys=True,
            ),
            transforms.RandFlipd(
                keys=["image"],
                prob=0.1,
                spatial_axis=0,
                allow_missing_keys=True,
            ),
            transforms.RandFlipd(
                keys=["image"],
                prob=0.1,
                spatial_axis=1,
                allow_missing_keys=True,
            ),
            transforms.RandFlipd(
                keys=["image"],
                prob=0.1,
                spatial_axis=2,
                allow_missing_keys=True,
            ),
            transforms.RandShiftIntensityd(
                keys=["image"],
                offsets=0.1,
                prob=0.5,
                allow_missing_keys=True,
            ),
            transforms.ToTensord(
                keys=["image"],
                allow_missing_keys=True,
            ),
        ])
        if not reshape:
            trans.transforms.append(
                transforms.RandGaussianSmoothd(
                    keys=["image"],
                    sigma_x=(0.5, 1.0),
                    sigma_y=(0.5, 1.0),
                    sigma_z=(0.5, 1.0),
                    prob=0.2,
                    allow_missing_keys=True,
                )
            )
    elif mode == 'test':
        trans = transforms.Compose([
            transforms.CastToTyped(
                keys=["image"],
                dtype=np.float32,
                allow_missing_keys=True,
            ),
            transforms.ToTensord(
                keys=["image"],
                allow_missing_keys=True,
            ),
        ])
    else:
        raise NotImplementedError(f"{mode} mode not implemented.")

    return trans


def vit_transforms(config: Any, mode: str = 'train') -> transforms.Compose:
    """
    Define ViT transforms based on the mode.

    Args:
        config (Any): Configuration object.
        mode (str): Mode of operation ('train', 'val', 'test').

    Returns:
        transforms.Compose: Composed transforms.
    """
    if mode == 'train':
        trans = transforms.Compose([
            transforms.CastToTyped(
                keys=["image", "label"],
                dtype=np.float32,
                allow_missing_keys=True,
            ),
            transforms.RandFlipd(
                keys=["image", "label"],
                prob=0.1,
                spatial_axis=0,
                allow_missing_keys=True,
            ),
            transforms.RandFlipd(
                keys=["image", "label"],
                prob=0.1,
                spatial_axis=1,
                allow_missing_keys=True,
            ),
            transforms.RandFlipd(
                keys=["image", "label"],
                prob=0.1,
                spatial_axis=2,
                allow_missing_keys=True,
            ),
            transforms.RandShiftIntensityd(
                keys=["image", "label"],
                offsets=0.1,
                prob=0.5,
                allow_missing_keys=True,
            ),
            transforms.ToTensord(
                keys=["image", "label"],
                allow_missing_keys=True,
            ),
        ])
    elif mode in ['val', 'test']:
        trans = transforms.Compose([
            transforms.CastToTyped(
                keys=["image", "label"],
                dtype=np.float32,
                allow_missing_keys=True,
            ),
            transforms.ToTensord(
                keys=["image", "label"],
                allow_missing_keys=True,
            ),
        ])
    else:
        raise NotImplementedError(f"{mode} mode not implemented.")

    return trans
