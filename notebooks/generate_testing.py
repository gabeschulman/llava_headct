import sys
sys.path.append("/gpfs/scratch/rpg8343/llava_headct")

import argparse
from datetime import datetime
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch import nn
from src.model import LLaVAHeadCT
from src.config_handler import ModelConfig, DataLoaderHandler





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_name = "pretrain_config"

objective = "condition_classification"

rank = dist.get_rank() if dist.is_initialized() else 0

world_size = dist.get_world_size() if dist.is_initialized() else 1

model_config = ModelConfig(config_name)

model = LLaVAHeadCT(
    **model_config.encoder_config,
    **model_config.projector_config,
    **model_config.decoder_config,
    state_dict_path=model_config.model_state_dict_path,
).to(device)

checkpoint_path = "/gpfs/scratch/gs4342/llava_headct/checkpoints/condition_classification/best_model.pth"

best_checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(best_checkpoint["model_state_dict"])
model.eval()

model_unwrapped = model.module if hasattr(model, 'module') else model


#### Testing to print out the id values of important tokens #####
print("Tokenizer special tokens:")
print(f"  EOS token id: {model_unwrapped.decoder.tokenizer.eos_token_id}")
print(f"  PAD token id: {model_unwrapped.decoder.tokenizer.pad_token_id}")
print(f"  BOS token id: {model_unwrapped.decoder.tokenizer.bos_token_id}")
print(f"  Decoder vocab size: {len(model_unwrapped.decoder.tokenizer)}")
print(f"  Decoder model embedding vocab size: {model_unwrapped.decoder.model.get_input_embeddings().weight.shape[0]}")

###### Testing to check ! id (the one that's filling the rest of the tokenization)
exclamation_id = model.decoder.tokenizer.convert_tokens_to_ids('!')
print("Exclamation mark token ID:", exclamation_id)


data_loader_handler = DataLoaderHandler(
    objective, model_config, rank=rank, world_size=world_size
)

test_dataloader = data_loader_handler.get_test_dataloader()

prompt_input_ids, prompt_attention_mask = data_loader_handler.get_objective_prompt_tokens(model, device)

#Hard Coded for Now
classification_prompt = "Describe the medical conditions observed in the attached head CT scan. Please list the conditions present in the following format: 'Conditions: condition 1, condition 2, ... condition N'. If no abnormalities are observed, please respond with 'Conditions: none.'"

count = 0

with torch.no_grad():
    for batch in test_dataloader:
        if batch is None:
            continue

        images = batch["image"].to(device, non_blocking=True)

        generated_ids = model.generate(images, prompt=classification_prompt, max_new_tokens=256)

        generated_texts = model.decoder.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        for gt_text, pred_text in zip(batch["objective"], generated_texts):
            print(f"Ground Truth: {gt_text}")
            print(f"Prediction: {pred_text}")
            print("-" * 40)

        count += 1
        if count > 10:
            break