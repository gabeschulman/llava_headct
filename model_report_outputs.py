import torch
import torch.distributed as dist
from src.model import LLaVAHeadCT
from src.config_handler import ModelConfig, DataLoaderHandler
from tqdm import tqdm
import json
import os

MAX_BATCHES_FOR_TEST = float("inf")  # Set to float('inf') to ignore
OUTPUT_DIR = "./generation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_name = "narrative_train_config"
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

data_loader_handler = DataLoaderHandler(
    model_config=model_config, rank=rank, world_size=world_size
)
test_dataloader = data_loader_handler.get_test_dataloader()

NARRATIVE_PROMPT = "Generate a detailed radiologist's medical narrative based on the findings from the attached head CT scan.\n\n"

generated_results = []

print(f"Rank {rank}: Starting Inference...")

with torch.no_grad():
    for batch_count, batch in enumerate(
        tqdm(test_dataloader, desc=f"Rank {rank} Eval"), 1
    ):
        if batch_count > MAX_BATCHES_FOR_TEST:
            print(
                f"\n*** Rank {rank}: Stopping after {MAX_BATCHES_FOR_TEST} batches for testing purposes. ***"
            )
            break

        if batch is None:
            continue

        images_batch = batch["image"].to(device, non_blocking=True)

        acc_nums_list = [str(acc_num) for acc_num in batch["accession_numbers"]]

        current_batch_size = len(acc_nums_list)

        batched_prompts = [NARRATIVE_PROMPT] * current_batch_size

        generated_ids = model.generate(
            images_batch, prompt=batched_prompts, max_new_tokens=512
        )
        generated_texts = model.decoder.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        for acc_num, response in zip(acc_nums_list, generated_texts):
            generated_results.append(
                {"accession_num": acc_num, "generated_report": response.strip()}
            )

output_file = os.path.join(OUTPUT_DIR, f"narratives_rank_{rank}.json")
print(f"Rank {rank}: Saving {len(generated_results)} reports to {output_file}")

with open(output_file, "w") as f:
    json.dump(generated_results, f, indent=4)

print(f"Rank {rank}: Finished.")
