"""
Pre-process images in chunks for parallel job submission.
Splits dataset into N chunks and processes each chunk independently.
"""

import argparse
import traceback
from typing import List, Tuple, Optional
import polars as pl
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from monai import transforms
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from multiprocessing import cpu_count

from src.fm_ct.data.transforms import MultipleWindowScaleStack


# Global transform pipeline (initialized once per worker)
_transform_pipeline = None


def _init_worker(roi, window_sizes, pixdim, axcodes, mode):
    """Initialize worker with transform pipeline."""
    global _transform_pipeline

    windowing_tran = MultipleWindowScaleStack(
        keys=["image"],
        window_sizes=window_sizes,
    )

    _transform_pipeline = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"], image_only=True),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.Orientationd(keys=["image"], axcodes=axcodes),
            transforms.Spacingd(keys=["image"], pixdim=pixdim, mode=mode),
            transforms.CropForegroundd(
                keys=["image"], source_key="image", allow_smaller=False
            ),
            transforms.Resized(keys=["image"], spatial_size=roi),
            windowing_tran,
            transforms.ToTensord(keys=["image"]),
        ]
    )


def process_single_image(args: Tuple[int, str, Path]) -> Tuple[int, Optional[str]]:
    """
    Process a single image using the pre-initialized transform pipeline.

    Returns:
        Tuple of (index, cached_path or None if error)
    """
    idx, img_path, output_path = args

    try:
        # Transform image using global pipeline
        sample = {"image": img_path}
        transformed = _transform_pipeline(sample)

        # Save preprocessed tensor (compressed)
        cache_filename = f"preprocessed_{idx:06d}.npy"
        cache_path = output_path / cache_filename

        # Save as numpy array (faster than torch.save)
        np.save(cache_path, transformed["image"].numpy())

        return (idx, str(cache_path))
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        traceback.print_exc()
        return (idx, None)


def preprocess_chunk(
    input_parquet: str,
    output_dir: str,
    chunk_id: int,
    total_chunks: int,
    roi: Tuple[int, int, int] = (96, 96, 96),
    window_sizes: List[Tuple[int, int]] = [(40, 80), (80, 200), (600, 2800)],
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    axcodes: str = "RAS",
    mode: int = 3,
    num_workers: int = None,
):
    """
    Pre-process a chunk of images.

    Args:
        input_parquet: Path to parquet file with image paths
        output_dir: Directory to save preprocessed images
        chunk_id: Which chunk to process (0-indexed)
        total_chunks: Total number of chunks
        roi: Region of interest size
        window_sizes: List of windowing parameters
        pixdim: Target pixel dimensions
        axcodes: Target orientation
        mode: Interpolation mode
        num_workers: Number of parallel workers (default: CPU count - 1)
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load full dataset
    df = pl.read_parquet(input_parquet)
    total_images = len(df)

    # Calculate chunk boundaries
    chunk_size = (total_images + total_chunks - 1) // total_chunks
    start_idx = chunk_id * chunk_size
    end_idx = min(start_idx + chunk_size, total_images)

    # Get chunk
    df_chunk = df[start_idx:end_idx]
    image_paths = df_chunk["img_path"].to_list()

    print(f"\nChunk {chunk_id}/{total_chunks-1}:")
    print(f"  Processing images {start_idx} to {end_idx-1} ({len(image_paths)} images)")
    print(f"  Using {num_workers} workers")

    # Prepare arguments for each image
    args_list = [
        (start_idx + local_idx, img_path, output_path)
        for local_idx, img_path in enumerate(image_paths)
    ]

    # Process images in parallel with initialized workers
    cached_paths = [None] * len(image_paths)

    # Use ProcessPoolExecutor with initializer
    initializer_fn = partial(_init_worker, roi, window_sizes, pixdim, axcodes, mode)

    with ProcessPoolExecutor(
        max_workers=num_workers, initializer=initializer_fn
    ) as executor:
        futures = {
            executor.submit(process_single_image, args): args for args in args_list
        }

        with tqdm(total=len(args_list), desc=f"Chunk {chunk_id}") as pbar:
            for future in as_completed(futures):
                try:
                    idx, cache_path = future.result()
                    local_idx = idx - start_idx
                    cached_paths[local_idx] = cache_path
                except Exception as e:
                    print(f"Future failed: {e}")
                pbar.update(1)

    # Save chunk metadata
    chunk_metadata = {
        "chunk_id": chunk_id,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "cached_paths": cached_paths,
        "successful": sum(1 for p in cached_paths if p is not None),
        "total": len(cached_paths),
    }

    metadata_path = output_path / f"chunk_{chunk_id:03d}_metadata.pt"
    torch.save(chunk_metadata, metadata_path)

    print(f"\nChunk {chunk_id} complete!")
    print(
        f"Successfully processed: {chunk_metadata['successful']}/{chunk_metadata['total']} images"
    )
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-process and cache images in chunks"
    )
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument(
        "--output", required=True, help="Output directory for cached images"
    )
    parser.add_argument(
        "--chunk-id", type=int, required=True, help="Chunk ID to process (0-indexed)"
    )
    parser.add_argument(
        "--total-chunks", type=int, required=True, help="Total number of chunks"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1)",
    )

    args = parser.parse_args()

    preprocess_chunk(
        args.input,
        args.output,
        args.chunk_id,
        args.total_chunks,
        num_workers=args.num_workers,
    )
