import torch
import torch.distributed as dist
from src.model import LLaVAHeadCT
from src.config_handler import ModelConfig, DataLoaderHandler

# sys.path.append("/gpfs/scratch/rpg8343/llava_headct")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_name = "narrative_train_config"

objective = "impression_generation"

checkpoint_path = "/gpfs/scratch/gs4342/llava_headct/checkpoints/cached_weights/impression_generation_pretraining_1.pth"

rank = dist.get_rank() if dist.is_initialized() else 0

world_size = dist.get_world_size() if dist.is_initialized() else 1

model_config = ModelConfig(config_name)

model = LLaVAHeadCT(
    **model_config.encoder_config,
    **model_config.projector_config,
    **model_config.decoder_config,
    state_dict_path=checkpoint_path,
).to(device)

# best_checkpoint = torch.load(checkpoint_path, map_location=device)
# model.load_state_dict(best_checkpoint["model_state_dict"])
model.eval()

model_unwrapped = model.module if hasattr(model, "module") else model


#### Testing to print out the id values of important tokens #####
print("Tokenizer special tokens:")
print(f"  EOS token id: {model_unwrapped.decoder.tokenizer.eos_token_id}")
print(f"  PAD token id: {model_unwrapped.decoder.tokenizer.pad_token_id}")
print(f"  BOS token id: {model_unwrapped.decoder.tokenizer.bos_token_id}")
print(f"  Decoder vocab size: {len(model_unwrapped.decoder.tokenizer)}")
print(
    f"  Decoder model embedding vocab size: {model_unwrapped.decoder.model.get_input_embeddings().weight.shape[0]}"
)

###### Testing to check ! id (the one that's filling the rest of the tokenization)
exclamation_id = model.decoder.tokenizer.convert_tokens_to_ids("!")
print("Exclamation mark token ID:", exclamation_id)


data_loader_handler = DataLoaderHandler(
    objective, model_config, rank=rank, world_size=world_size
)

test_dataloader = data_loader_handler.get_test_dataloader()

(
    prompt_input_ids,
    prompt_attention_mask,
) = data_loader_handler.get_objective_prompt_tokens(model, device)

# Hard Coded for Now
classification_prompt = "Generate a detailed radiologist's medical impression based on the findings from the attached head CT scan."

count = 0

with torch.no_grad():
    for batch in test_dataloader:
        if batch is None:
            continue

        images = batch["image"].to(device, non_blocking=True)

        generated_ids = model.generate(
            images, prompt=classification_prompt, max_new_tokens=256
        )

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
