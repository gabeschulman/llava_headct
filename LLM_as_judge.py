# GT: "/gpfs/scratch/gs4342/data/cached_images/test/nyu_test_processed_cached.parquet"
# Generated: "/gpfs/scratch/rpg8343/data/gen_out/generated_reports.json"
# To identify: ["cancer","hydrocephalus","edema","dementia","intraparenchymal hemorrhage","intraventricular hemorrhage","subdural hematoma","epidural hematoma","subarachnoid hemmorrhage","intracerebral hemorrhage","fracture","hematoma",]


import pandas as pd
import json
import torch
import re
from transformers import pipeline
from tqdm import tqdm

# ==========================================
# 1. SETUP AND CONFIGURATION
# ==========================================

GT_PATH = (
    "/gpfs/scratch/gs4342/data/cached_images/test/nyu_test_processed_cached.parquet"
)
GEN_PATH = "/gpfs/scratch/rpg8343/data/gen_out/generated_reports.json"
OUTPUT_PATH = "evaluation_vectors_full.json"  # Path to save final results

ITERATIONS_MAX = float("inf")

IDENTIFIERS = [
    "cancer",
    "hydrocephalus",
    "edema",
    "dementia",
    "intraparenchymal hemorrhage",
    "intraventricular hemorrhage",
    "subdural hematoma",
    "epidural hematoma",
    "subarachnoid hemmorrhage",
    "intracerebral hemorrhage",
    "fracture",
    "hematoma",
]

MODEL_ID = "Qwen/Qwen1.5-7B-Chat"

# ==========================================
# 2. DATA LOADING AND PREPROCESSING
# ==========================================

print("--- Loading Data ---")

# Load Ground Truth
try:
    df_gt = pd.read_parquet(GT_PATH)
    # create the full text
    df_gt["gt_report"] = (
        df_gt["impression_deid"].fillna("") + " " + df_gt["narrative_deid"].fillna("")
    )
    # Ensure accession_num is string for consistent merging
    df_gt["accession_num"] = df_gt["accession_num"].astype(str)
    print(f"Loaded {len(df_gt)} GT records.")
except Exception as e:
    print(f"Error loading Parquet: {e}")
    exit()

# Load Generated Reports
try:
    with open(GEN_PATH, "r") as f:
        gen_data = json.load(f)

    # Convert JSON list/dict to DataFrame
    # Assuming gen_data is a list of dicts like [{'accession_num': '123', 'generated_report': '...'}, ...]
    if isinstance(gen_data, list):
        df_gen = pd.DataFrame(gen_data)
    elif isinstance(gen_data, dict):
        # If it's a dict where keys are accession numbers
        # Adjust logic based on actual structure. Assuming list of dicts based on prompt context.
        df_gen = pd.DataFrame.from_dict(gen_data, orient="index")
        df_gen["accession_num"] = df_gen.index

    df_gen["accession_num"] = df_gen["accession_num"].astype(str)
    print(f"Loaded {len(df_gen)} Generated records.")
except Exception as e:
    print(f"Error loading JSON: {e}")
    exit()

# Merge Dataframes on accession_num
# We use an inner join to only process scans where we have BOTH GT and Gen reports
merged_df = pd.merge(
    df_gt[["accession_num", "gt_report"]],
    df_gen[["accession_num", "generated_report"]],
    on="accession_num",
    how="inner",
)

# Sort by accession_num as requested
merged_df = merged_df.sort_values(by="accession_num").reset_index(drop=True)

print(f"Processing {len(merged_df)} matched records.")

# ==========================================
# 3. MODEL INITIALIZATION
# ==========================================

print("--- Initializing Model ---")
try:
    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# ==========================================
# 4. PROMPT ENGINEERING & PARSING UTILS
# ==========================================


def create_prompt(report_text, identifiers):
    """
    Constructs the prompt to force Qwen to output a binary vector.
    """
    id_list_str = ", ".join(identifiers)

    system_prompt = (
        "You are an expert radiologist assistant. Your task is to read a radiology report "
        "and determine if specific medical conditions are present."
    )

    user_prompt = (
        f"Read the following radiology report:\n\n"
        f'"""{report_text}"""\n\n'
        f"For each of the following conditions, determine if it is affirmatively indicated as present in the report:\n"
        f"[{id_list_str}]\n\n"
        "Return the result STRICTLY as a JSON list of 0s and 1s corresponding to the order of the conditions above. "
        "Use 1 for present, 0 for absent/negated/not mentioned. "
        "Do not output any explanation, markdown, or other text. Just the list.\n"
        "Example Output: [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_output(output_text):
    """
    Extracts the list of integers from the model response.
    """
    try:
        # 1. Attempt to find a list structure [0, 1, ...]
        match = re.search(r"\[([01,\s]+)\]", output_text)
        if match:
            # Extract content, split by comma, convert to int
            vector_str = match.group(1)
            vector = [
                int(x.strip()) for x in vector_str.split(",") if x.strip().isdigit()
            ]

            # Validation: Vector length must match identifiers length
            if len(vector) == len(IDENTIFIERS):
                return vector

        # Fallback: if no brackets found, look for sequence of 0s and 1s
        # This is riskier but helpful if model forgets brackets
        fallback = re.findall(r"\b[01]\b", output_text)
        if len(fallback) == len(IDENTIFIERS):
            return [int(x) for x in fallback]

        print(f"Warning: Could not parse valid vector from: {output_text[:50]}...")
        return None  # parsing failed

    except Exception:
        return None


# ==========================================
# 5. INFERENCE LOOP
# ==========================================

results = {}

print("--- Starting Inference ---")

# Iterate through the dataframe
# Note: For massive datasets, you might batch this differently,
# but a loop is clearest for per-accession tracking.
num_iterations = 0
for index, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
    num_iterations += 1
    acc_num = row["accession_num"]
    gt_text = row["gt_report"]
    gen_text = row["generated_report"]

    # --- Process Ground Truth ---
    messages_gt = create_prompt(gt_text, IDENTIFIERS)

    # We strip the prompt from the output to save memory and clean parsing
    out_gt = pipe(messages_gt, max_new_tokens=128, return_full_text=False)
    # return_full_text=False ensures we only get the generated response part
    gt_response_text = out_gt[0]["generated_text"]
    gt_vector = parse_output(gt_response_text)

    # --- Process Generated Report ---
    messages_gen = create_prompt(gen_text, IDENTIFIERS)

    out_gen = pipe(messages_gen, max_new_tokens=128, return_full_text=False)
    gen_response_text = out_gen[0]["generated_text"]
    gen_vector = parse_output(gen_response_text)

    # --- Store Result ---
    # Only store if both parses were successful (or handle Nones as you see fit)
    results[acc_num] = {
        "gt_vector": gt_vector if gt_vector else "PARSE_ERROR",
        "gen_vector": gen_vector if gen_vector else "PARSE_ERROR",
        "gt_raw": gt_response_text if not gt_vector else None,  # Debug info if fail
        "gen_raw": gen_response_text if not gen_vector else None,
    }
    if num_iterations > ITERATIONS_MAX:
        print("\n--- Hit max iterations ---")
        break

# ==========================================
# 6. OUTPUT AND SAVE
# ==========================================

print("\n--- Inference Complete. Formatting Output ---")

# Convert results dict to a sorted list of dictionaries for final output
final_output = []
sorted_keys = sorted(results.keys())

for acc in sorted_keys:
    data = results[acc]
    final_output.append(
        {
            "accession_num": acc,
            "gt_vector": data["gt_vector"],
            "gen_vector": data["gen_vector"],
        }
    )

# Save to JSON
with open(OUTPUT_PATH, "w") as f:
    json.dump(final_output, f, indent=4)

print(f"Results saved to {OUTPUT_PATH}")
print(f"Identifiers order: {IDENTIFIERS}")
