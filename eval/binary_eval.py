import torch
import torch.distributed as dist
from src.model import LLaVAHeadCT
from src.config_handler import ModelConfig, DataLoaderHandler
import pandas as pd
from tqdm import tqdm


MAX_BATCHES_FOR_TEST = float(
    "inf"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_name = "narrative_train_config"
objective = "narrative_generation"
checkpoint_path = (
    "/gpfs/scratch/gs4342/llava_headct/checkpoints/14558456/best_model.pth"
)

rank = dist.get_rank() if dist.is_initialized() else 0
world_size = dist.get_world_size() if dist.is_initialized() else 1

model_config = ModelConfig(config_name)
model_config.dataloader_config["batch_size"] = 16
model_config.dataloader_config["num_workers"] = 8
model_config.dataloader_config["persistent_workers"] = True

model = LLaVAHeadCT(
    **model_config.encoder_config,
    **model_config.projector_config,
    **model_config.decoder_config,
    state_dict_path=checkpoint_path,
).to(device)
model.eval()

model_unwrapped = model.module if hasattr(model, "module") else model

data_loader_handler = DataLoaderHandler(
    model_config=model_config, rank=rank, world_size=world_size
)

test_dataloader = data_loader_handler.get_test_dataloader()

parquet_path = (
    "/gpfs/scratch/gs4342/data/cached_images/test/nyu_test_processed_cached.parquet"
)
pf = pd.read_parquet(parquet_path)
pf = pf.drop_duplicates(subset="accession_num")

pf["accession_num"] = pf["accession_num"].astype(str)
pf = pf.set_index("accession_num")

binary_columns = [
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
C = len(binary_columns)

identifier_names = {
    "cancer": "cancer",
    "hydrocephalus": "hydrocephalus",
    "edema": "edema",
    "dementia": "dementia",
    "IPH": "intraparenchymal hemorrhage",
    "IVH": "intraventricular hemorrhage",
    "SDH": "subdural hematoma",
    "EDH": "epidural hematoma",
    "SAH": "subarachnoid hemmorrhage",
    "ICH": "intracerebral hemorrhage",
    "fracture": "fracture",
    "hematoma": "hematoma",
}

metrics = {}
ratio = {}
ratio["Total"] = {"Yes": 0, "No": 0}
metrics["NoMatch"] = 0
for cond in binary_columns:
    metrics[cond] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    ratio[cond] = {"Yes": 0, "No": 0}


condition_prompts = []
for curr_cond in binary_columns:
    name = identifier_names[curr_cond]
    condition_prompts.append(
        f"For the following individual condition, indicate whether it is present ('Yes') or absent ('No'): {name}. Respond with only: Yes or No\n\nANSWER:"
    )


with torch.no_grad():
    for batch_count, batch in enumerate(tqdm(test_dataloader, desc="Evaluation"), 1):
        if batch_count > MAX_BATCHES_FOR_TEST:
            print(
                f"\n*** Stopping after {MAX_BATCHES_FOR_TEST} batches for testing purposes. ***"
            )
            break

        if batch is None:
            print(f"WARNING: Batch {batch_count} is None. Skipping.")
            continue

        full_images_batch = batch["image"].to(device, non_blocking=True)

        acc_nums_list = [str(acc_num) for acc_num in batch["accession_numbers"]]

        try:
            batch_df = pf.loc[acc_nums_list]
        except KeyError:
            valid_acc_nums_series = pf.index.intersection(acc_nums_list)
            if valid_acc_nums_series.empty:
                continue
            batch_df = pf.loc[valid_acc_nums_series]

        valid_acc_nums = batch_df.index.tolist()

        batch_rows = batch_df.T.to_dict("series")

        valid_indices = []
        acc_num_to_original_index = {
            acc_num: i for i, acc_num in enumerate(acc_nums_list)
        }

        for acc_num in valid_acc_nums:
            valid_indices.append(acc_num_to_original_index[acc_num])

        if not valid_indices:
            continue

        B_valid = len(valid_indices)

        valid_images_batch = full_images_batch[valid_indices]
        final_image_batch = torch.repeat_interleave(
            valid_images_batch, repeats=C, dim=0
        )
        batched_prompts = condition_prompts * B_valid

        generated_ids = model.generate(
            final_image_batch, prompt=batched_prompts, max_new_tokens=256
        )
        generated_texts = model.decoder.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        for idx, response in enumerate(generated_texts):
            image_idx = idx // C
            condition_idx = idx % C
            acc_num_str = valid_acc_nums[image_idx]
            curr_cond = binary_columns[condition_idx]
            curr_row = batch_rows[acc_num_str]
            condition_present = curr_row[curr_cond]
            gt = "Yes" if condition_present else "No"
            response = response.strip().lower().capitalize()

            if condition_present:
                ratio["Total"]["Yes"] += 1
                ratio[curr_cond]["Yes"] += 1
            else:
                ratio["Total"]["No"] += 1
                ratio[curr_cond]["No"] += 1

            if response != "Yes" and response != "No":
                metrics["NoMatch"] += 1

            elif response == "Yes" and gt == "Yes":
                metrics[curr_cond]["TP"] += 1
            elif response == "No" and gt == "Yes":
                metrics[curr_cond]["FN"] += 1
            elif response == "No" and gt == "No":
                metrics[curr_cond]["TN"] += 1
            elif response == "Yes" and gt == "No":
                metrics[curr_cond]["FP"] += 1


print("\n Per Condition:")
tp_sum = 0
tn_sum = 0
fp_sum = 0
fn_sum = 0
for condition in metrics.keys():
    if condition == "NoMatch":
        print(f"No Match: {metrics[condition]}")
    else:
        print(f"{condition.capitalize()}:")
        q = ratio[condition]["Yes"] / (ratio[condition]["Yes"] + ratio[condition]["No"])

        tp = metrics[condition]["TP"]
        tp_sum += tp

        tn = metrics[condition]["TN"]
        tn_sum += tn

        fp = metrics[condition]["FP"]
        fp_sum += fp

        fn = metrics[condition]["FN"]
        fn_sum += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else -1
        recall = tp / (tp + fn) if (tp + fn) > 0 else -1
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        print(f"    Precision: {round(precision,4)}") if precision != -1 else print(
            "    Precision: Not enough relevant data"
        )
        print(f"    Recall: {round(recall,4)}") if recall != -1 else print(
            "    Recall: Not enough relevant data"
        )
        print(f"    Accuracy: {round(accuracy,4)}\n")

        print("Expected w.r.t. ratio:")
        print(f"Precision: {round(q, 4)}")
        print(f"Recall: {round(q, 4)}")
        print(f"Accuracy: {round(q*q + (1 - q)*(1 - q), 4)}\n")

yes_count = ratio["Total"]["Yes"]
no_count = ratio["Total"]["No"]
yes_ratio = yes_count / (yes_count + no_count)

total_precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else -1
total_recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else -1
total_accuracy = (
    (tp_sum + tn_sum) / (tp_sum + tn_sum + fp_sum + fn_sum)
    if (tp_sum + tn_sum + fp_sum + fn_sum) > 0
    else 0.0
)

print("\n Complete Metrics:")
print(f"    Yes to No Ratio: {yes_count}:{no_count}")
print(
    f"    Total Precision: {round(total_precision,4)}"
) if total_precision != -1 else print("    Total Precision: Not enough relevant data")
print(f"    Total Recall: {round(total_recall,4)}") if total_recall != -1 else print(
    "    Total Recall: Not enough relevant data"
)
print(f"    Total Accuracy: {round(total_accuracy,4)}")

print("\nExpected w.r.t. ratio:")
print(f"Precision: {round(yes_ratio, 4)}")
print(f"Recall: {round(yes_ratio, 4)}")
print(f"Accuracy: {round(yes_ratio*yes_ratio + (1 - yes_ratio)*(1 - yes_ratio), 4)}")

print("\nExpected with 50/50 guessing:")
print(f"Precision: {round(yes_ratio, 4)}")
print("Recall: 0.5")
print("Accuracy: 0.5")
