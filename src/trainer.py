from datetime import datetime
import json
import os
from pathlib import Path
import torch
from torch import nn
from src.logger import setup_logging
from src.model import LLaVAHeadCT
from src.dataloader import create_condition_classification_dataloader


def main():
    logger = setup_logging()
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"GPU count: {torch.cuda.device_count()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    config_path = Path(__file__).parent / "configs" / "pretrain_config.json"
    logger.info(f"Loading config from: {config_path}")
    with open(config_path) as f:
        config = json.load(f)
    logger.info(f"Loaded config: {config}")

    logger.info("Initializing model...")
    model = LLaVAHeadCT(
        vision_encoder_weights=config["vision_encoder"]["vision_encoder_weights"],
        vision_encoder_in_chans=config["vision_encoder"]["vision_encoder_in_chans"],
        vision_encoder_img_size=config["vision_encoder"]["vision_encoder_img_size"],
        vision_encoder_patch_size=config["vision_encoder"]["vision_encoder_patch_size"],
        projector_input_channels=config["projector"]["input_channels"],
        projector_inner_channels=config["projector"]["inner_channels"],
        projector_out_channels=config["projector"]["out_channels"],
        decoder_model_name=config["decoder"]["model_name"],
        projector_dropout=config["training"].get("projector_dropout", 0.0),
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    logger.info("Setting up dataloader...")
    train_file = os.path.join(
        config["dataset"]["datapath"], config["dataset"]["processed_files"]["train"]
    )
    batch_size = config["training"]["batch_size"]
    dataloader = create_condition_classification_dataloader(train_file, batch_size)
    logger.info(f"Dataset size: {len(dataloader.dataset)}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Total batches: {len(dataloader)}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"].get("base_lr", 1e-4),
        weight_decay=config["training"].get("weight_decay", 0.04),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    num_epochs = config.get("num_epochs", 10)

    logger.info(f"Training for {num_epochs} epochs")
    logger.info(f"Learning rate: {config['training'].get('base_lr', 1e-4)}")
    logger.info(f"Weight decay: {config['training'].get('weight_decay', 0.04)}")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        epoch_start_time = datetime.now()

        logger.info(f"Starting epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            condition_texts = batch["condition"]

            target_tokens = model.decoder.tokenizer(
                condition_texts, return_tensors="pt", padding=True, truncation=True
            )
            target_ids = target_tokens["input_ids"].to(device)
            outputs = model(images)

            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            vocab_size = shift_logits.size(-1)
            loss = criterion(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                )

            if batch_idx % 100 == 0 and batch_idx > 0:
                checkpoint_path = (
                    f"checkpoints/checkpoint_epoch_{epoch+1}_batch_{batch_idx}.pth"
                )
                Path("checkpoints").mkdir(exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item(),
                    },
                    checkpoint_path,
                )
                logger.info(f"Saved checkpoint: {checkpoint_path}")

    avg_loss = total_loss / len(dataloader)
    epoch_time = datetime.now() - epoch_start_time
    logger.info(
        f"Epoch {epoch} completed, Average Loss: {avg_loss:.4f}, Time Taken: {epoch_time}"
    )

    final_model_path = f"checkpoints/final_model_epoch_{num_epochs}.pth"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model: {final_model_path}")


if __name__ == "__main__":
    main()
