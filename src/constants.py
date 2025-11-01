import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATAFILES = {
    "datapath": "/gpfs/data/razavianlab/data/capstone_f25/flat_files",
    "image_files": {
        "test": "nyu_test_label.csv",
        "train": "nyu_train_label.csv",
        "val": "nyu_val_label.csv",
    },
    "radiology_report": "/gpfs/data/razavianlab/capstone/2025_stroke/ct_accession_report_2506_deid.csv",
    "processed_files": {
        "test": "nyu_test_processed.parquet",
        "train": "nyu_train_processed.parquet",
        "val": "nyu_val_processed.parquet",
    },
}
