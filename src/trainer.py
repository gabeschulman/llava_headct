from datetime import datetime
import json
import os
from pathlib import Path
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from src.logger import setup_logging
from src.model import LLaVAHeadCT
from src.dataloader import create_condition_classification_dataloader


def validate(model, dataloader, criterion, device, use_amp=True):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            target_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                outputs = model(
                    images, input_ids=target_ids, attention_mask=attention_mask
                )

                num_image_tokens = 513
                text_logits = outputs.logits[:, num_image_tokens:, :].contiguous()

                shift_logits = text_logits[:, :-1, :].contiguous()
                shift_labels = target_ids[:, 1:].contiguous()

                vocab_size = shift_logits.size(-1)
                loss = criterion(
                    shift_logits.view(-1, vocab_size), shift_labels.view(-1)
                )

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


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
        projector_dropout=config["training"].get("projector_dropout", 0.1),
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    logger.info("Setting up train dataloader...")
    train_file = os.path.join(
        config["dataset"]["cached_images_dir"],
        config["dataset"]["cached_files"]["train"],
    )
    batch_size = config["training"]["batch_size"]
    gradient_accumulation_steps = config["training"].get(
        "gradient_accumulation_steps", 1
    )
    num_workers = config["training"].get("num_workers", 4)
    train_dataloader = create_condition_classification_dataloader(
        train_file,
        batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        tokenizer_model_name=config["decoder"]["model_name"],
        max_text_length=512,
        use_cached_images=True,
    )
    logger.info(f"Train dataset size: {len(train_dataloader.dataset)}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    logger.info(f"Total train batches: {len(train_dataloader)}")

    logger.info("Setting up validation dataloader...")
    val_file = os.path.join(
        config["dataset"]["cached_images_dir"], config["dataset"]["cached_files"]["val"]
    )
    val_dataloader = create_condition_classification_dataloader(
        val_file,
        batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        tokenizer_model_name=config["decoder"]["model_name"],
        max_text_length=512,
        use_cached_images=True,
    )
    logger.info(f"Val dataset size: {len(val_dataloader.dataset)}")
    logger.info(f"Total val batches: {len(val_dataloader)}")
    logger.info(f"Num workers: {num_workers}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"].get("base_lr", 1e-4),
        weight_decay=config["training"].get("weight_decay", 0.04),
    )

    scaler = GradScaler()
    use_amp = config["training"].get("use_amp", True)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    num_epochs = config.get("num_epochs", 10)

    logger.info(f"Training for {num_epochs} epochs")
    logger.info(f"Learning rate: {config['training'].get('base_lr', 1e-4)}")
    logger.info(f"Weight decay: {config['training'].get('weight_decay', 0.04)}")
    logger.info(f"Mixed precision (AMP): {use_amp}")

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        epoch_start_time = datetime.now()

        logger.info(f"Starting epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(train_dataloader):
            images = batch["image"].to(device, non_blocking=True)
            target_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                outputs = model(
                    images, input_ids=target_ids, attention_mask=attention_mask
                )

                num_image_tokens = 513
                text_logits = outputs.logits[:, num_image_tokens:, :].contiguous()

                shift_logits = text_logits[:, :-1, :].contiguous()
                shift_labels = target_ids[:, 1:].contiguous()

                vocab_size = shift_logits.size(-1)
                loss = criterion(
                    shift_logits.view(-1, vocab_size), shift_labels.view(-1)
                )
                loss = loss / gradient_accumulation_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * gradient_accumulation_steps

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}, Batch {batch_idx+1}, Train Loss: {loss.item():.4f}"
                )

            if batch_idx % 100 == 0 and batch_idx > 0:
                checkpoint_path = (
                    f"checkpoints/checkpoint_epoch_{epoch+1}_batch_{batch_idx+1}.pth"
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

        # Calculate average training loss
        avg_train_loss = total_train_loss / len(train_dataloader)
        epoch_time = datetime.now() - epoch_start_time

        # Run validation
        logger.info(f"Running validation for epoch {epoch+1}...")
        val_loss = validate(model, val_dataloader, criterion, device, use_amp)

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} completed - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Time: {epoch_time}"
        )

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_path = "checkpoints/best_model.pth"
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                },
                best_model_path,
            )
            logger.info(f"New best model saved! Val Loss: {val_loss:.4f}")

        # Save epoch checkpoint
        epoch_checkpoint_path = f"checkpoints/epoch_{epoch+1}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
            },
            epoch_checkpoint_path,
        )

    logger.info(
        f"Training completed! Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}"
    )
    logger.info("Best model saved at: checkpoints/best_model.pth")


if __name__ == "__main__":
    main()
