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

PRETRAINED_VIT_WEIGHTS: str = "/gpfs/scratch/gs4342/llava_headct/checkpoints/vit_pretrained/vit_base_patch12_96.pth"

PROMPT_TEMPLATES: dict[str, str] = {
    "narrative_generation": "Generate a detailed radiologist's medical narrative based on the findings from the attached head CT scan.\n\n",
    "impression_generation": "Provide a concise radiologist's medical impression based on the findings from the attached head CT scan.\n\n",
    "condition_classification": "List all medical conditions present. Format your response as: CONDITIONS: [list] or CONDITIONS: none\n\nCONDITIONS:",
    "individual_condition_classification": "For the following individual condition, indicate whether it is present ('Yes') or absent ('No'): {condition}. Respond with only: Yes or No\n\nANSWER:",
}

# Phase 1
# OBJECTIVE_DICT = {
#     "narrative_generation": 0.025,
#     "condition_classification": 0.7,
#     "individual_condition_classification": 0.275,
#     "impression_generation": 0.0,
# }

# OBJECTIVE_SCALES = {
#     "condition_classification": 3.0,
#     "individual_condition_classification": 2.0,
#     "narrative_generation": 1.0,
#     "impression_generation": 0.25,
# }

# Phase 2
OBJECTIVE_SCALES = {
    "narrative_generation": 1.0,
    "condition_classification": 3.0,
    "individual_condition_classification": 2.0,
    "impression_generation": 0.5,
}

OBJECTIVE_DICT = {
    "narrative_generation": 0.35,
    "condition_classification": 0.35,
    "individual_condition_classification": 0.25,
    "impression_generation": 0.05,
}

ABBREVIATED_CONDITIONS_DICT: dict = {
    "IPH": "intraparenchymal hemorrhage",
    "IVH": "intraventricular hemorrhage",
    "SDH": "subdural hemorrhage",
    "EDH": "epidural hemorrhage",
    "SAH": "subarachnoid hemorrhage",
    "ICH": "intracerebral hemorrhage",
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
