import argparse
from datetime import datetime
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from src.logger import setup_logging
from src.model import LLaVAHeadCT
from src.config_handler import ModelConfig, DataLoaderHandler


def is_main_process():
    """Check if current process is the main process (rank 0 or not distributed)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def compute_loss(outputs, target_ids, criterion, num_image_tokens=513, prompt_length=0):
    """
    Compute the cross-entropy loss for text generation with teacher forcing.

    With teacher forcing:
    - Input to model: [image] + [prompt] + [target[:-1]]
    - Model outputs logits for: [image positions] + [prompt positions] + [target positions (shifted)]
    - We compute loss only on: target token predictions

    Args:
        outputs: Model outputs with logits
        target_ids: Target token IDs [batch, target_len]
        criterion: Loss function
        num_image_tokens: Number of image tokens to skip
        prompt_length: Length of prompt tokens (to skip when computing loss)

    Returns:
        loss: Computed loss value
    """
    logits_for_targets = outputs.logits[
        :, num_image_tokens + prompt_length - 1 : -1, :
    ].contiguous()
    labels = target_ids[:, 1:].contiguous()

    vocab_size = logits_for_targets.size(-1)

    loss = criterion(logits_for_targets.view(-1, vocab_size), labels.view(-1))

    del logits_for_targets, labels

    return loss


def validate(
    model, prompt_ids, prompt_attention, dataloader, criterion, device, use_amp=True
):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue

            images = batch["image"].to(device, non_blocking=True)
            target_ids = batch["input_ids"].to(device, non_blocking=True)

            current_batch_size = images.shape[0]
            prompt_ids_batch = prompt_ids.expand(current_batch_size, -1)
            prompt_attention_batch = prompt_attention.expand(current_batch_size, -1)
            prompt_length = prompt_ids_batch.shape[1]

            full_input_ids = torch.cat([prompt_ids_batch, target_ids[:, :-1]], dim=1)
            target_attention_mask = torch.ones_like(
                target_ids[:, :-1], device=target_ids.device
            )
            full_attention_mask = torch.cat(
                [prompt_attention_batch, target_attention_mask], dim=1
            )

            with autocast(enabled=use_amp):
                outputs = model(
                    images, input_ids=full_input_ids, attention_mask=full_attention_mask
                )
                loss = compute_loss(
                    outputs,
                    target_ids,
                    criterion,
                    num_image_tokens=513,
                    prompt_length=prompt_length,
                )

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main(objective: str, config_name: str):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    logger = setup_logging()
    if is_main_process():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        if world_size > 1:
            logger.info(f"Distributed training: world_size={world_size}, rank={rank}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_main_process():
        logger.info(f"Using device: {device}")

    model_config = ModelConfig(config_name)
    if is_main_process():
        logger.info(f"Loaded config: {model_config}")
    data_loader_handler = DataLoaderHandler(
        objective, model_config, rank=rank, world_size=world_size
    )

    logger.info("Initializing model...")
    model: LLaVAHeadCT | torch.nn.parallel.DistributedDataParallel = LLaVAHeadCT(
        **model_config.encoder_config,
        **model_config.projector_config,
        **model_config.decoder_config,
    )
    model.to(device)

    # Wrap model in DistributedDataParallel for multi-GPU training
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], output_device=rank, find_unused_parameters=False
        )
        logger.info(f"Model wrapped in DDP for rank {rank}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    logger.info("Setting up train dataloader...")
    batch_size = model_config.dataloader_config["batch_size"]
    gradient_accumulation_steps = model_config.train_config.get(
        "gradient_accumulation_steps", 1
    )
    num_workers = model_config.train_config.get("num_workers", 4)
    train_dataloader: DataLoader = data_loader_handler.get_train_dataloader()
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    logger.info(f"Total train batches: {len(train_dataloader)}")

    logger.info("Setting up validation dataloader...")
    val_dataloader = data_loader_handler.get_val_dataloader()
    logger.info(f"Total val batches: {len(val_dataloader)}")
    logger.info(f"Num workers: {num_workers}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_config.train_config.get("base_lr", 1e-4),
        weight_decay=model_config.train_config.get("weight_decay", 0.04),
    )

    scaler = GradScaler()
    use_amp = model_config.train_config.get("use_amp", True)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    num_epochs = model_config.train_config["num_epochs"]

    logger.info(f"Training for {num_epochs} epochs")
    logger.info(f"Learning rate: {model_config.train_config['base_lr']}")
    logger.info(f"Weight decay: {model_config.train_config['weight_decay']}")
    logger.info(f"Mixed precision (AMP): {use_amp}")

    best_val_loss = float("inf")
    best_epoch = 0

    # Get unwrapped model for tokenizer access
    model_unwrapped = model.module if world_size > 1 else model
    prompt_tokens: tuple = data_loader_handler.get_objective_prompt_tokens(
        model_unwrapped,  # type: ignore
        device,
    )
    prompt_input_ids: torch.Tensor = prompt_tokens[0]
    prompt_attention_mask: torch.Tensor = prompt_tokens[1]

    for epoch in range(num_epochs):
        if world_size > 1:
            train_dataloader.sampler.set_epoch(epoch)  # type: ignore
        model.train()
        total_train_loss = 0.0
        epoch_start_time = datetime.now()

        logger.info(f"Starting epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(train_dataloader):
            if batch is None:
                continue

            images = batch["image"].to(device, non_blocking=True)
            target_ids = batch["input_ids"].to(device, non_blocking=True)

            batch_size_current = images.shape[0]
            prompt_ids_batch = prompt_input_ids.expand(batch_size_current, -1)
            prompt_mask_batch = prompt_attention_mask.expand(batch_size_current, -1)
            prompt_length = prompt_ids_batch.shape[1]

            full_input_ids = torch.cat([prompt_ids_batch, target_ids[:, :-1]], dim=1)
            target_attention_mask = torch.ones_like(
                target_ids[:, :-1], device=target_ids.device
            )
            full_attention_mask = torch.cat(
                [prompt_mask_batch, target_attention_mask], dim=1
            )

            with autocast(enabled=use_amp):
                outputs = model(
                    images, input_ids=full_input_ids, attention_mask=full_attention_mask
                )
                loss = compute_loss(
                    outputs,
                    target_ids,
                    criterion,
                    num_image_tokens=513,
                    prompt_length=prompt_length,
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
                if is_main_process():
                    logger.info(
                        f"Epoch {epoch+1}, Batch {batch_idx+1}, Train Loss: {loss.item():.4f}"
                    )

            if batch_idx % 1000 == 0 and batch_idx > 0:
                if is_main_process():
                    checkpoint_path = f"checkpoints/{objective}/checkpoint_epoch_{epoch+1}_batch_{batch_idx+1}.pth"
                    Path(f"checkpoints/{objective}").mkdir(exist_ok=True, parents=True)
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

        avg_train_loss = total_train_loss / len(train_dataloader)
        epoch_time = datetime.now() - epoch_start_time

        if is_main_process():
            logger.info(f"Running validation for epoch {epoch+1}...")
        val_loss = validate(
            model,
            prompt_input_ids,
            prompt_attention_mask,
            val_dataloader,
            criterion,
            device,
            use_amp,
        )

        if is_main_process():
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} completed - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Time: {epoch_time}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            if is_main_process():
                best_model_path = f"checkpoints/{objective}/best_model.pth"
                Path(f"checkpoints/{objective}").mkdir(exist_ok=True, parents=True)
                # Unwrap DDP model for saving
                model_to_save = model.module if world_size > 1 else model
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model_to_save.state_dict(),  # type: ignore
                        "optimizer_state_dict": optimizer.state_dict(),  # type: ignore
                        "train_loss": avg_train_loss,
                        "val_loss": val_loss,
                    },
                    best_model_path,
                )
                logger.info(f"New best model saved! Val Loss: {val_loss:.4f}")

        if is_main_process():
            epoch_checkpoint_path = f"checkpoints/{objective}/epoch_{epoch+1}.pth"
            model_to_save = model.module if world_size > 1 else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),  # type: ignore
                    "optimizer_state_dict": optimizer.state_dict(),  # type: ignore
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                },
                epoch_checkpoint_path,
            )

    if is_main_process():
        logger.info(
            f"Training completed! Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}"
        )
        logger.info(f"Best model saved at: checkpoints/{objective}/best_model.pth")

        logger.info("=" * 80)
        logger.info("Running final evaluation on test set...")
        logger.info("Loading best model checkpoint...")

    best_checkpoint = torch.load(f"checkpoints/{objective}/best_model.pth")
    model_unwrapped = model.module if world_size > 1 else model
    model_unwrapped.load_state_dict(best_checkpoint["model_state_dict"])  # type: ignore

    if is_main_process():
        logger.info("Setting up test dataloader...")
    test_dataloader = data_loader_handler.get_test_dataloader()
    if is_main_process():
        logger.info(f"Total test batches: {len(test_dataloader)}")

    test_loss = validate(
        model,
        prompt_input_ids,
        prompt_attention_mask,
        test_dataloader,
        criterion,
        device,
        use_amp,
    )

    if is_main_process():
        logger.info("=" * 80)
        logger.info("FINAL TEST SET RESULTS:")
        logger.info(f"  Test Loss: {test_loss:.4f}")
        logger.info(f"  Best Validation Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
        logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--objective", type=str, help="Objective column name")
    parser.add_argument("--config_name", type=str, help="Config file name")
    args = parser.parse_args()

    main(objective=args.objective, config_name=args.config_name)
