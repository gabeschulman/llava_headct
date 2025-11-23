import os
from typing import List

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

PROMPT_TEMPLATES: dict[str, str] = {
    "condition_classification": "Describe the medical conditions observed in the attached head CT scan. Please list the conditions present in the following format: 'Conditions: condition 1, condition 2, ... condition N'. If no abnormalities are observed, please respond with 'Conditions: none.'",
    "impression_generation": "Provide a concise radiologist's medical impression based on the findings from the attached head CT scan.",
    "narrative_generation": "Generate a detailed radiologist's medical narrative based on the findings from the attached head CT scan.",
    "individual_condition_classification": "For the following individual condition, indicate whether it is present ('Yes') or not ('No') based on the attached head CT scan: {condition}.",
}

OBJECTIVE_DICT = {
    "condition_classification": 0.3,
    "narrative_generation": 0.1,
    "impression_generation": 0.2,
    "individual_condition_classification": 0.4,
    # "q_and_a": 0.1,
}

INDIVIDUAL_CONDITIONS_LIST: List[str] = [
    "cancer",
    "hydrocephalus",
    "edema",
    "dementia",
    "IPH",
    "IVH",
    "SDH",
    "EDH",
    "SAH",
    "ICH",
    "fracture",
    "hematoma",
]
