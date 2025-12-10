"""
Script to merge Q&A data from multiple radiology_qa_final JSON files into existing parquet files.

This script:
1. Loads and combines Q&A data from three JSON files
2. For each accession number, includes up to 3 Q&A pairs
3. Joins with existing parquet files on accession_num (INNER JOIN - only keeps rows with Q&A)
4. Adds q1, a1, q2, a2, q3, a3 columns (nulls if fewer than 3 Q&As)
5. Saves filtered parquet files with only Q&A samples

Usage:
    python merge_qa_to_parquet_combined.py
"""

import json
from pathlib import Path
import polars as pl
from typing import List, Dict

# Configuration
QA_JSON_PATHS = [
    "/gpfs/scratch/bm3772/radiology_qa_final_part1.json",
    "/gpfs/scratch/bm3772/radiology_qa_final_part2.json",
    "/gpfs/scratch/bm3772/radiology_qa_final_part3.json",
]
PARQUET_BASE_PATH = "/gpfs/scratch/gs4342/data/cached_images"
OUTPUT_BASE_PATH = "/gpfs/scratch/bm3772/data/all_questions"

SPLITS = ["train", "val", "test"]
PARQUET_FILES = {
    "train": "train/nyu_train_processed_cached.parquet",
    "val": "val/nyu_val_processed_cached.parquet",
    "test": "test/nyu_test_processed_cached.parquet",
}

def load_and_combine_qa_data(qa_json_paths: List[str]) -> Dict:
    """
    Load Q&A data from multiple JSON files and combine them.
    
    Returns:
        Dictionary mapping accession_num to list of Q&A items
    """
    print("Loading Q&A data from multiple files...")
    combined_qa = {}
    
    for json_path in qa_json_paths:
        print(f"  Loading {json_path}...")
        try:
            with open(json_path, "r") as f:
                qa_data = json.load(f)
            
            print(f"    Loaded {len(qa_data)} entries")
            
            # Combine Q&A data by accession number
            for entry in qa_data:
                accession_num = entry.get("accession_num")
                qa_items = entry.get("qa", {}).get("items", [])
                
                if not accession_num or not qa_items:
                    continue
                
                if accession_num not in combined_qa:
                    combined_qa[accession_num] = []
                
                combined_qa[accession_num].extend(qa_items)
        
        except Exception as e:
            print(f"    ERROR loading {json_path}: {e}")
            continue
    
    print(f"\nCombined Q&A data for {len(combined_qa)} unique accession numbers")
    
    # Print some statistics
    qa_counts = [len(items) for items in combined_qa.values()]
    print(f"  Total Q&A pairs: {sum(qa_counts)}")
    print(f"  Avg Q&A pairs per accession: {sum(qa_counts) / len(qa_counts):.2f}")
    print(f"  Max Q&A pairs for one accession: {max(qa_counts)}")
    print(f"  Min Q&A pairs for one accession: {min(qa_counts)}")
    
    return combined_qa


def create_qa_lookup(combined_qa: Dict) -> pl.DataFrame:
    """
    Create a polars DataFrame with up to 3 Q&A pairs per accession number.
    
    Returns:
        DataFrame with columns: accession_num, q1, a1, q2, a2, q3, a3
    """
    print("\nCreating Q&A lookup table with up to 3 Q&A pairs per accession...")
    
    qa_records = []
    
    for accession_num, qa_items in combined_qa.items():
        # Take first 3 Q&A pairs (no randomization)
        selected_qas = qa_items[:3]
        
        record = {"accession_num": accession_num}
        
        # Add Q&A pairs (with nulls if fewer than 3)
        for i in range(3):
            if i < len(selected_qas):
                record[f"q{i+1}"] = selected_qas[i]["q"]
                record[f"a{i+1}"] = selected_qas[i]["a"]
            else:
                record[f"q{i+1}"] = None
                record[f"a{i+1}"] = None
        
        qa_records.append(record)
    
    # Create DataFrame with explicit schema to handle nulls properly
    schema = {
        "accession_num": pl.Utf8,
        "q1": pl.Utf8,
        "a1": pl.Utf8,
        "q2": pl.Utf8,
        "a2": pl.Utf8,
        "q3": pl.Utf8,
        "a3": pl.Utf8,
    }
    
    qa_df = pl.DataFrame(qa_records, schema=schema)
    print(f"Created Q&A lookup with {len(qa_df)} entries")
    
    # Show distribution of how many Q&A pairs each accession has
    num_with_1 = qa_df.filter(pl.col("q1").is_not_null() & pl.col("q2").is_null()).height
    num_with_2 = qa_df.filter(pl.col("q2").is_not_null() & pl.col("q3").is_null()).height
    num_with_3 = qa_df.filter(pl.col("q3").is_not_null()).height
    
    print(f"  Accessions with 1 Q&A pair: {num_with_1}")
    print(f"  Accessions with 2 Q&A pairs: {num_with_2}")
    print(f"  Accessions with 3 Q&A pairs: {num_with_3}")
    
    print("\nSample Q&A records:")
    print(qa_df.head(2))
    
    return qa_df


def merge_qa_with_parquet(parquet_path: str, qa_df: pl.DataFrame, output_path: str):
    """
    Load a parquet file, join with Q&A data, and save the result.
    
    Args:
        parquet_path: Path to input parquet file
        qa_df: DataFrame with Q&A data (accession_num, q1, a1, q2, a2, q3, a3)
        output_path: Path to save the merged parquet file
    """
    print(f"\nProcessing {parquet_path}...")
    
    # Load existing parquet
    df = pl.read_parquet(parquet_path)
    print(f"  Original shape: {df.shape}")
    print(f"  Original columns: {df.columns}")
    
    # Join with Q&A data (inner join to ONLY keep rows with Q&A data)
    merged_df = df.join(qa_df, on="accession_num", how="inner")
    print(f"  Merged shape: {merged_df.shape}")
    print(f"  Kept only rows with Q&A data: {merged_df.shape[0]} rows")
    
    # Show how many have 1, 2, or 3 Q&A pairs in this split
    num_with_1 = merged_df.filter(pl.col("q1").is_not_null() & pl.col("q2").is_null()).height
    num_with_2 = merged_df.filter(pl.col("q2").is_not_null() & pl.col("q3").is_null()).height
    num_with_3 = merged_df.filter(pl.col("q3").is_not_null()).height
    print(f"  Split distribution:")
    print(f"    1 Q&A pair: {num_with_1}")
    print(f"    2 Q&A pairs: {num_with_2}")
    print(f"    3 Q&A pairs: {num_with_3}")
    
    # Create output directory if it doesn't exist
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Save merged parquet
    merged_df.write_parquet(output_path)
    print(f"  Saved to {output_path}")
    
    # Show sample of merged data
    print("  Sample of merged data:")
    sample = merged_df.select(["accession_num", "q1", "a1", "q2", "a2"]).head(2)
    print(sample)


def main():
    """Main function to merge Q&A data with all parquet files."""
    
    print("=" * 80)
    print("COMBINING Q&A DATA FROM MULTIPLE FILES")
    print("=" * 80)
    
    # Load and combine Q&A data from all three files
    combined_qa = load_and_combine_qa_data(QA_JSON_PATHS)
    
    # Create Q&A lookup table (up to 3 Q&A pairs per accession number)
    qa_df = create_qa_lookup(combined_qa)
    
    print("\n" + "=" * 80)
    print("MERGING WITH PARQUET FILES")
    print("=" * 80)
    
    # Process each split
    for split in SPLITS:
        input_path = Path(PARQUET_BASE_PATH) / PARQUET_FILES[split]
        output_path = Path(OUTPUT_BASE_PATH) / PARQUET_FILES[split]
        
        merge_qa_with_parquet(str(input_path), qa_df, str(output_path))
    
    print("\n" + "=" * 80)
    print("✓ MERGE COMPLETE!")
    print("=" * 80)
    print(f"Output location: {OUTPUT_BASE_PATH}")
    print(f"\nDirectory structure:")
    print(f"  {OUTPUT_BASE_PATH}/")
    print(f"    ├── train/nyu_train_processed_cached.parquet")
    print(f"    ├── val/nyu_val_processed_cached.parquet")
    print(f"    └── test/nyu_test_processed_cached.parquet")
    
    print("\nColumns in output files:")
    print("  - All original columns from cached_images")
    print("  - q1, a1 (first Q&A pair)")
    print("  - q2, a2 (second Q&A pair, null if not available)")
    print("  - q3, a3 (third Q&A pair, null if not available)")
    
    print("\nNext steps:")
    print("1. Verify the merged files look correct")
    print("2. Update your training config to use this new data path")
    print("3. Modify trainer to randomly select from available Q&A pairs during training")
    print("=" * 80)


if __name__ == "__main__":
    main()