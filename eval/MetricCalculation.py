import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

INPUT_FILE = "evaluation_vectors_10k.json" 
OUTPUT_CSV = "LLM_AS_JUDGE_EVAL_10k.csv"

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

print(f"--- Loading data from {INPUT_FILE} ---")

try:
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

y_true = []
y_pred = []
valid_accessions = []

parse_errors = 0

for entry in data:
    gt = entry.get("gt_vector")
    gen = entry.get("gen_vector")

    # Filter out any failed parses from the previous step
    if gt == "PARSE_ERROR" or gen == "PARSE_ERROR" or gt is None or gen is None:
        parse_errors += 1
        continue

    y_true.append(gt)
    y_pred.append(gen)
    valid_accessions.append(entry["accession_num"])

print(f"Total Records: {len(data)}")
print(f"Valid Pairs for Eval: {len(y_true)}")
print(f"Parse Errors Skipped: {parse_errors}")

Y_true = np.array(y_true)
Y_pred = np.array(y_pred)

print("\n--- Calculating Breakdown by Identifier ---")

results_list = []

total_tp = 0
total_tn = 0
total_fp = 0
total_fn = 0

for i, identifier in enumerate(IDENTIFIERS):
    yt = Y_true[:, i]
    yp = Y_pred[:, i]

    tp = np.sum((yt == 1) & (yp == 1))
    tn = np.sum((yt == 0) & (yp == 0))
    fp = np.sum((yt == 0) & (yp == 1))
    fn = np.sum((yt == 1) & (yp == 0))

    total_tp += tp
    total_tn += tn
    total_fp += fp
    total_fn += fn

    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    if (tp + fp + tn + fn) > 0:
        acc = (tp + tn) / (tp + tn + fp + fn)
    else:
        acc = 0.0

    results_list.append(
        {
            "Identifier": identifier,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": acc,
        }
    )

if (total_tp + total_fp) > 0:
    overall_precision = total_tp / (total_tp + total_fp)
else:
    overall_precision = 0.0

if (total_tp + total_fn) > 0:
    overall_recall = total_tp / (total_tp + total_fn)
else:
    overall_recall = 0.0

if (overall_precision + overall_recall) > 0:
    overall_f1 = (
        2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
    )
else:
    overall_f1 = 0.0

if (total_tp + total_fp + total_tn + total_fn) > 0:
    overall_acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
else:
    overall_acc = 0.0

results_list.append(
    {
        "Identifier": "OVERALL",
        "TP": total_tp,
        "TN": total_tn,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": overall_precision,
        "Recall": overall_recall,
        "F1": overall_f1,
        "Accuracy": overall_acc,
    }
)

df_results = pd.DataFrame(results_list)
exact_matches = accuracy_score(Y_true, Y_pred)

print("\n--- Performance Report ---")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

print(df_results.round(3).to_string(index=False))

print(f"\nExact Match Ratio (All identifiers correct in a report): {exact_matches:.4f}")

df_results.to_csv(OUTPUT_CSV, index=False)
print(f"\nMetrics saved to: {OUTPUT_CSV}")
