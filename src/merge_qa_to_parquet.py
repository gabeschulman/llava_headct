"""
Script to merge Q&A data from radiology_qa_final.json into existing parquet files.

This script:
1. Loads the Q&A JSON file
2. For each accession number, randomly selects one Q&A pair
3. Joins with existing parquet files on accession_num (INNER JOIN - only keeps rows with Q&A)
4. Adds qa_question and qa_answer columns
5. Saves filtered parquet files with only Q&A samples

Usage:
    python merge_qa_to_parquet.py
"""

import json
import random
from pathlib import Path
import polars as pl

# Configuration
QA_JSON_PATH = "/gpfs/scratch/bm3772/radiology_qa_final.json"
PARQUET_BASE_PATH = "/gpfs/scratch/gs4342/data/cached_images"
OUTPUT_BASE_PATH = (
    "/gpfs/scratch/bm3772/data/cached_images_qa_only"  # New directory for output
)

SPLITS = ["train", "val", "test"]
PARQUET_FILES = {
    "train": "train/nyu_train_processed_cached.parquet",
    "val": "val/nyu_val_processed_cached.parquet",
    "test": "test/nyu_test_processed_cached.parquet",
}

# Set random seed for reproducibility
random.seed(42)


def load_qa_data(qa_json_path: str) -> dict:
    """Load Q&A data from JSON file."""
    print(f"Loading Q&A data from {qa_json_path}...")
    with open(qa_json_path, "r") as f:
        qa_data = json.load(f)
    print(f"Loaded {len(qa_data)} Q&A entries")
    return qa_data


def create_qa_lookup(qa_data: list) -> pl.DataFrame:
    """
    Create a polars DataFrame with one randomly selected Q&A pair per accession number.

    Returns:
        DataFrame with columns: accession_num, qa_question, qa_answer
    """
    print("Creating Q&A lookup table...")

    qa_records = []

    for entry in qa_data:
        accession_num = entry.get("accession_num")
        qa_items = entry.get("qa", {}).get("items", [])

        if not accession_num or not qa_items:
            continue

        # Randomly select one Q&A pair
        selected_qa = random.choice(qa_items)

        qa_records.append(
            {
                "accession_num": accession_num,
                "qa_question": selected_qa["q"],
                "qa_answer": selected_qa["a"],
            }
        )

    qa_df = pl.DataFrame(qa_records)
    print(f"Created Q&A lookup with {len(qa_df)} entries")
    print("Sample Q&A pairs:")
    print(qa_df.head(3))

    return qa_df


def merge_qa_with_parquet(parquet_path: str, qa_df: pl.DataFrame, output_path: str):
    """
    Load a parquet file, join with Q&A data, and save the result.

    Args:
        parquet_path: Path to input parquet file
        qa_df: DataFrame with Q&A data (accession_num, qa_question, qa_answer)
        output_path: Path to save the merged parquet file
    """
    print(f"\nProcessing {parquet_path}...")

    # Load existing parquet
    df = pl.read_parquet(parquet_path)
    print(f"  Original shape: {df.shape}")
    print(f"  Columns: {df.columns}")

    # Join with Q&A data (inner join to ONLY keep rows with Q&A data)
    merged_df = df.join(qa_df, on="accession_num", how="inner")
    print(f"  Merged shape: {merged_df.shape}")
    print(f"  Kept only rows with Q&A data: {merged_df.shape[0]} rows")

    # Create output directory if it doesn't exist
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Save merged parquet
    merged_df.write_parquet(output_path)
    print(f"  Saved to {output_path}")

    # Show sample of merged data
    print("  Sample of merged data:")
    sample = merged_df.select(["accession_num", "qa_question", "qa_answer"]).head(2)
    print(sample)


def main():
    """Main function to merge Q&A data with all parquet files."""

    # Load Q&A data
    qa_data = load_qa_data(QA_JSON_PATH)

    # Create Q&A lookup table (one Q&A pair per accession number)
    qa_df = create_qa_lookup(qa_data)

    # Process each split
    for split in SPLITS:
        input_path = Path(PARQUET_BASE_PATH) / PARQUET_FILES[split]
        output_path = Path(OUTPUT_BASE_PATH) / PARQUET_FILES[split]

        merge_qa_with_parquet(str(input_path), qa_df, str(output_path))

    print("\n" + "=" * 80)
    print("✓ Merge complete!")
    print(f"Updated parquet files saved to: {OUTPUT_BASE_PATH}")
    print("=" * 80)

    print("\nNext steps:")
    print("1. Verify the merged files look correct")
    print("2. Create a new config file (e.g., qa_train_config.json)")
    print("3. Add 'qa_training' to supported_objectives in config_handler.py")
    print("4. Modify the trainer to handle variable prompts per sample")


if __name__ == "__main__":
    main()
