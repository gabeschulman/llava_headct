"""
Merge cached chunks and create final parquet file.
Run this after all chunk jobs have completed.
"""

import argparse
import polars as pl
import torch
from pathlib import Path


def merge_chunks(
    input_parquet: str,
    cache_dir: str,
    total_chunks: int,
) -> str:
    """
    Merge cached chunks into a single parquet file with cached paths.

    Args:
        input_parquet: Original parquet file
        cache_dir: Directory where cached images were saved
        total_chunks: Total number of chunks that were processed

    Returns:
        Path to merged parquet file
    """
    cache_path = Path(cache_dir)

    # Load original dataframe
    df = pl.read_parquet(input_parquet)
    total_images = len(df)

    print(f"Merging {total_chunks} chunks...")
    print(f"Total images in dataset: {total_images}")

    # Collect cached paths from all chunks
    all_cached_paths = [None] * total_images
    total_successful = 0

    for chunk_id in range(total_chunks):
        metadata_path = cache_path / f"chunk_{chunk_id:03d}_metadata.pt"

        if not metadata_path.exists():
            print(f"WARNING: Chunk {chunk_id} metadata not found at {metadata_path}")
            continue

        metadata = torch.load(metadata_path)
        start_idx = metadata["start_idx"]
        end_idx = metadata["end_idx"]
        cached_paths = metadata["cached_paths"]
        successful = metadata["successful"]

        print(
            f"  Chunk {chunk_id}: {successful}/{len(cached_paths)} successful "
            f"(indices {start_idx}-{end_idx-1})"
        )

        # Merge cached paths
        for local_idx, cache_path_str in enumerate(cached_paths):
            global_idx = start_idx + local_idx
            all_cached_paths[global_idx] = cache_path_str
            if cache_path_str is not None:
                total_successful += 1

    # Add cached paths to dataframe
    df_cached = df.with_columns([pl.Series("cached_path", all_cached_paths)])

    # Save merged parquet
    output_parquet = cache_path / f"{Path(input_parquet).stem}_cached.parquet"
    df_cached.write_parquet(output_parquet)

    print("\nMerging complete!")
    print(
        f"Successfully cached: {total_successful}/{total_images} images "
        f"({100*total_successful/total_images:.1f}%)"
    )
    print(f"Merged parquet saved to: {output_parquet}")

    return str(output_parquet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge cached chunks")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument(
        "--cache-dir", required=True, help="Directory with cached images"
    )
    parser.add_argument(
        "--total-chunks", type=int, required=True, help="Total number of chunks"
    )

    args = parser.parse_args()

    merge_chunks(args.input, args.cache_dir, args.total_chunks)
