import argparse
from datetime import datetime
from pathlib import Path
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch import nn
from torch.cuda.amp import autocast
from src.logger import setup_logging
from src.model import LLaVAHeadCT
from src.config_handler import ModelConfig, DataLoaderHandler
from src.train_utils import determine_is_resume, get_training_weight_config


def is_main_process():
    """Check if current process is the main process (rank 0 or not distributed)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def compute_teacher_forcing_loss(
    outputs,
    target_ids,
    num_image_tokens=513,
    prompt_length=0,
    pad_token_id=-100,
    reduction="mean",
):
    """
    Compute the cross-entropy loss for text generation with teacher forcing.

    With teacher forcing:
    - Input to model: [image] + [prompt] + [target[:-1]]
    - Model outputs logits for: [image positions] + [prompt positions] + [target positions (shifted)]
    - We compute loss only on: target token predictions

    Args:
        outputs: Model outputs with logits
        target_ids: Target token IDs [batch, target_len]
        num_image_tokens: Number of image tokens to skip
        prompt_length: Length of prompt tokens (to skip when computing loss)

    Returns:
        loss: Computed loss value
    """
    logits_for_targets = outputs.logits[
        :, num_image_tokens + prompt_length - 1 :
    ].contiguous()
    labels = target_ids.contiguous()
    labels = torch.where(
        labels == pad_token_id,
        torch.tensor(-100, device=labels.device),
        labels,
    )

    vocab_size = logits_for_targets.size(-1)
    batch_size = logits_for_targets.size(0)
    seq_len = logits_for_targets.size(1)

    criterion_per_token = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    loss_per_token = criterion_per_token(
        logits_for_targets.view(-1, vocab_size), labels.view(-1)
    )
    if reduction == "none":
        loss_per_token = loss_per_token.view(batch_size, seq_len)
        valid_mask = (labels != -100).float()
        valid_counts = valid_mask.sum(dim=1).clamp(min=1)
        loss_per_sample = (loss_per_token * valid_mask).sum(dim=1) / valid_counts
        return loss_per_sample
    else:
        return loss_per_token.mean()


def compute_contrastive_loss(
    image_embeddings: torch.Tensor,
    text_embeddings: list[torch.Tensor],
    temperature=0.07,
    reduction="mean",
):
    """
    Compute contrastive loss between image and text embeddings.

    Args:
        image_embeddings: Image embeddings tensor of shape [batch_size, embed_dim]
        text_embeddings: List of text embeddings tensors, each of shape [n_tokens, embed_dim]
        temperature: Scaling factor for logits (default is value for CLIP)
    Returns:
        loss: Computed contrastive loss
    """
    batch_size = image_embeddings.size(0)

    # Take embedding means
    image_embeddings = image_embeddings.mean(dim=1)
    text_embeddings = torch.stack([te.mean(dim=0) for te in text_embeddings])

    # Normalize embeddings
    image_embeddings = nn.functional.normalize(image_embeddings, p=2, dim=1)
    text_embeddings = nn.functional.normalize(text_embeddings, p=2, dim=1)

    # Compute similarity matrix
    logits = torch.matmul(image_embeddings, text_embeddings.t()) / temperature
    logits = torch.clamp(logits, min=-15, max=15)

    labels = torch.arange(batch_size).to(image_embeddings.device)

    criterion = nn.CrossEntropyLoss(reduction=reduction)
    loss_i2t = criterion(logits, labels)
    loss_t2i = criterion(logits.t(), labels)

    loss = (loss_i2t + loss_t2i) / 2.0

    return loss


def validate(
    model,
    dataloader,
    device,
    use_amp=True,
    pad_token_id=-100,
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

            image_embeddings = model.get_image_embeddings(images)

            prompt_ids_batch = batch["prompt_input_ids"].to(device, non_blocking=True)
            prompt_attention_batch = batch["prompt_attention_mask"].to(
                device, non_blocking=True
            )
            prompt_length = prompt_ids_batch.shape[1]

            full_input_ids = torch.cat([prompt_ids_batch, target_ids[:, :-1]], dim=1)
            target_attention_mask = torch.ones_like(
                target_ids[:, :-1], device=target_ids.device
            )
            full_attention_mask = torch.cat(
                [prompt_attention_batch, target_attention_mask], dim=1
            )

            contrastive_input_ids = batch["contrastive_input_ids"]
            contrastive_text_embeddings = [
                model.get_text_embeddings(input_ids.to(device, non_blocking=True))
                for input_ids in contrastive_input_ids
            ]

            sample_weights = batch["sample_weights"].to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                outputs = model(
                    image_embeddings=image_embeddings,
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                )
                tf_loss = compute_teacher_forcing_loss(
                    outputs,
                    target_ids,
                    num_image_tokens=image_embeddings.shape[1],
                    prompt_length=prompt_length,
                    pad_token_id=pad_token_id,
                    reduction="none",
                )
                contrastive_loss = compute_contrastive_loss(
                    image_embeddings, contrastive_text_embeddings, reduction="none"
                )
                weighted_tf_loss = (tf_loss * sample_weights).mean()
                weighted_cont_loss = (contrastive_loss * sample_weights).mean()
                loss = weighted_tf_loss + weighted_cont_loss

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main(job_id: int, config_name: str):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    main_process_flag = is_main_process()
    is_resume = determine_is_resume(config_name)

    logger = setup_logging()
    if main_process_flag:
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        if world_size > 1:
            logger.info(f"Distributed training: world_size={world_size}, rank={rank}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if main_process_flag:
        logger.info(f"Using device: {device}")

    model_config = ModelConfig(config_name)
    if main_process_flag:
        logger.info(f"Loaded config: {model_config}")
    data_loader_handler = DataLoaderHandler(
        model_config, rank=rank, world_size=world_size
    )

    if model_config.model_state_dict_path and main_process_flag:
        logger.info(
            f"Loading model state dict from: {model_config.model_state_dict_path}"
        )
        checkpoint = torch.load(model_config.model_state_dict_path, map_location=device)

    logger.info("Initializing model...")
    model: LLaVAHeadCT | torch.nn.parallel.DistributedDataParallel = LLaVAHeadCT(
        **model_config.encoder_config,
        **model_config.projector_config,
        **model_config.decoder_config,
        state_dict_path=model_config.model_state_dict_path,
    )
    model.to(device)
    pad_token_id = (
        model.decoder.tokenizer.pad_token_id if model.decoder.tokenizer else -100
    )
    logger.info(f"Pad token ID: {pad_token_id}")

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
    seed = model_config.dataloader_config.get("seed")
    logger.info(f"Using seed: {seed}")
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

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": model_config.train_config.get("encoder_lr", 1e-6),
            },
            {
                "params": model.projector.parameters(),
                "lr": model_config.train_config.get("projector_lr", 1e-4),
            },
            {
                "params": model.decoder.parameters(),
                "lr": model_config.train_config.get("decoder_lr", 5e-5),
            },
        ],
        weight_decay=model_config.train_config.get("weight_decay", 0.01),
    )

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=1000 // gradient_accumulation_steps,
    )

    constant_scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=1.0,
        total_iters=1000000,
    )

    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, constant_scheduler], milestones=[1000]
    )

    scaler = torch.cuda.amp.GradScaler(
        init_scale=2.0**12,
        growth_interval=2000,
        backoff_factor=0.5,
        growth_factor=2.0,
    )
    use_amp = model_config.train_config.get("use_amp", True)
    if use_amp:
        scaler = torch.cuda.amp.GradScaler(
            init_scale=2.0**12,
            growth_interval=2000,
            backoff_factor=0.5,
            growth_factor=2.0,
        )

        if (
            "scaler_state_dict" in checkpoint
            and checkpoint["scaler_state_dict"] is not None
        ):
            try:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
                logger.info(f"✓ Loaded scaler state (scale={scaler.get_scale()})")
            except Exception as e:
                logger.warning(f"⚠ Failed to load scaler state: {e}")
                logger.warning(
                    f"  Using conservative init (scale={scaler.get_scale()})"
                )
        else:
            logger.info("No scaler state in checkpoint")
            logger.info(f"Using conservative init (scale={scaler.get_scale()})")

    num_epochs = model_config.train_config["num_epochs"]

    logger.info(f"Training for {num_epochs} epochs")
    logger.info(f"Learning rate: {model_config.train_config['base_lr']}")
    logger.info(f"Weight decay: {model_config.train_config['weight_decay']}")
    logger.info(f"Mixed precision (AMP): {use_amp}")

    best_val_loss = float("inf")
    best_epoch = 0

    model_unwrapped = model.module if world_size > 1 else model

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

            image_embeddings = model.get_image_embeddings(images)

            prompt_ids_batch = batch["prompt_input_ids"].to(device, non_blocking=True)
            prompt_mask_batch = batch["prompt_attention_mask"].to(
                device, non_blocking=True
            )
            prompt_length = prompt_ids_batch.shape[1]

            full_input_ids = torch.cat([prompt_ids_batch, target_ids[:, :-1]], dim=1)
            target_attention_mask = torch.ones_like(
                target_ids[:, :-1], device=target_ids.device
            )
            full_attention_mask = torch.cat(
                [prompt_mask_batch, target_attention_mask], dim=1
            )

            contrastive_input_ids = batch["contrastive_input_ids"]
            contrastive_text_embeddings = [
                model.get_text_embeddings(input_ids.to(device, non_blocking=True))
                for input_ids in contrastive_input_ids
            ]

            sample_weights = batch["sample_weights"].to(device, non_blocking=True)
            objective_scales = batch["objective_scales"].to(device, non_blocking=True)
            global_batch_idx = epoch * len(train_dataloader) + batch_idx

            sample_weights_adjusted, contrastive_weight = get_training_weight_config(
                epoch, global_batch_idx, sample_weights, is_resume=is_resume
            )

            with autocast(enabled=use_amp):
                outputs = model(
                    image_embeddings=image_embeddings,
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                )
                tf_loss = compute_teacher_forcing_loss(
                    outputs,
                    target_ids,
                    num_image_tokens=image_embeddings.shape[1],
                    prompt_length=prompt_length,
                    pad_token_id=pad_token_id,
                    reduction="none",
                )
                contrastive_loss = compute_contrastive_loss(
                    image_embeddings, contrastive_text_embeddings, reduction="none"
                )
                weighted_tf_loss = (
                    tf_loss * objective_scales * sample_weights_adjusted
                ).mean()
                weighted_cont_loss = (contrastive_loss * sample_weights_adjusted).mean()
                loss = weighted_tf_loss + contrastive_weight * weighted_cont_loss
                loss = loss / gradient_accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(
                    f"Rank {torch.distributed.get_rank()}, Batch {batch_idx}: Invalid loss, skipping backward"
                )
                logger.warning(f"Objective text: {batch['objective'][0]}")
                optimizer.zero_grad()
                continue

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                has_nan_grad = any(
                    torch.isnan(p.grad).any()
                    for p in model.parameters()
                    if p.grad is not None
                )
                if has_nan_grad:
                    logger.warning(f"NaN after unscaling at batch {batch_idx}")
                    logger.warning(f"Accession numbers: {batch['accession_number']}")
                    optimizer.zero_grad()
                    scaler.update()
                    continue
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            total_train_loss += loss.item() * gradient_accumulation_steps

            if batch_idx % 10 == 0 and main_process_flag:
                logger.info(
                    f"Epoch {epoch+1}, Batch {batch_idx+1}, Train Loss: {loss.item():.4f}"
                )

            if (
                global_batch_idx % 25000 == 0
                and global_batch_idx > 0
                and main_process_flag
            ):
                logger.info(f"\n=== Batch {global_batch_idx} Training Dynamics ===")
                logger.info(f"\n=== PROMPT CHECK (Batch {batch_idx}) ===")
                logger.info(f"Objective type: {batch['objective_type'][0]}")
                logger.info(f"Full prompt: [{batch['prompt'][0]}]")
                logger.info(f"Prompt ends with: [{batch['prompt'][0][-30:]}]")
                logger.info(f"Target: [{batch['objective'][0]}]")
                logger.info("===\n")
                logger.info(f"Contrastive weight: {contrastive_weight}")
                logger.info(f"Raw TF loss mean: {tf_loss.mean().item():.4f}")
                logger.info(f"Objective scales: {objective_scales.tolist()}")
                logger.info(f"Weighted TF loss: {weighted_tf_loss.item():.4f}")
                logger.info(f"Weighted cont loss: {weighted_cont_loss.item():.4f}")
                logger.info(f"Objectives in batch: {batch['objective_type']}")
                logger.info(
                    f"Objective counts: {dict((x, batch['objective_type'].count(x)) for x in set(batch['objective_type']))}"
                )
                logger.info("Sample prompts:")
                for i in range(min(2, len(batch["prompt"]))):
                    logger.info(
                        f"  [{batch['objective_type'][i]}] {batch['prompt'][i][:80]}..."
                    )
                logger.info("Sample targets:")
                for i in range(min(2, len(batch["objective"]))):
                    logger.info(
                        f"  [{batch['objective_type'][i]}] {batch['objective'][i][:80]}..."
                    )

            if batch_idx % 2500 == 0 and batch_idx > 0 and main_process_flag:
                checkpoint_path = f"checkpoints/{job_id}/checkpoint_epoch_{epoch+1}_batch_{batch_idx+1}.pth"
                Path(f"checkpoints/{job_id}").mkdir(exist_ok=True, parents=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item(),
                        "scaler_state_dict": scaler.state_dict() if use_amp else None,
                    },
                    checkpoint_path,
                )
                logger.info(f"Saved checkpoint: {checkpoint_path}")

            if (
                global_batch_idx % 25000 == 0
                and global_batch_idx > 0
                and main_process_flag
            ):
                count = 0
                test_dataloader = data_loader_handler.get_val_dataloader()
                check_prompt = "Generate a detailed radiologist's medical impression based on the findings from the attached head CT scan."
                model.eval()
                with torch.no_grad():
                    for batch in test_dataloader:
                        if batch is None:
                            continue

                        images = batch["image"].to(device, non_blocking=True)

                        generated_ids = model.generate(
                            images, prompt=check_prompt, max_new_tokens=256
                        )

                        generated_texts = model.decoder.tokenizer.batch_decode(
                            generated_ids, skip_special_tokens=True
                        )

                        for gt_text, pred_text in zip(
                            batch["objective"], generated_texts
                        ):
                            logger.info(f"Ground Truth: {gt_text}")
                            logger.info(f"Prediction: {pred_text}")
                            logger.info("-" * 40)

                        count += 1
                        if count > 2:
                            break

        avg_train_loss = total_train_loss / len(train_dataloader)
        epoch_time = datetime.now() - epoch_start_time

        if main_process_flag:
            logger.info(f"Running validation for epoch {epoch+1}...")
        val_loss = validate(
            model,
            val_dataloader,
            device,
            use_amp=use_amp,
            pad_token_id=pad_token_id,
        )

        if main_process_flag:
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} completed - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Time: {epoch_time}"
            )

        if val_loss < best_val_loss and main_process_flag:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_path = f"checkpoints/{job_id}/best_model.pth"
            Path(f"checkpoints/{job_id}").mkdir(exist_ok=True, parents=True)
            # Unwrap DDP model for saving
            model_to_save = model.module if world_size > 1 else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),  # type: ignore
                    "optimizer_state_dict": optimizer.state_dict(),  # type: ignore
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "scaler_state_dict": scaler.state_dict() if use_amp else None,
                },
                best_model_path,
            )
            logger.info(f"New best model saved! Val Loss: {val_loss:.4f}")

        if main_process_flag:
            epoch_checkpoint_path = f"checkpoints/{job_id}/epoch_{epoch+1}.pth"
            model_to_save = model.module if world_size > 1 else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),  # type: ignore
                    "optimizer_state_dict": optimizer.state_dict(),  # type: ignore
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "scaler_state_dict": scaler.state_dict() if use_amp else None,
                },
                epoch_checkpoint_path,
            )

    if main_process_flag:
        logger.info(
            f"Training completed! Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}"
        )
        logger.info(f"Best model saved at: checkpoints/{job_id}/best_model.pth")

        logger.info("=" * 80)
        logger.info("Running final evaluation on test set...")
        logger.info("Loading best model checkpoint...")

    best_checkpoint = torch.load(f"checkpoints/{job_id}/best_model.pth")
    model_unwrapped = model.module if world_size > 1 else model
    model_unwrapped.load_state_dict(best_checkpoint["model_state_dict"])  # type: ignore

    if main_process_flag:
        logger.info("Setting up test dataloader...")
    test_dataloader = data_loader_handler.get_test_dataloader()
    if main_process_flag:
        logger.info(f"Total test batches: {len(test_dataloader)}")

    test_loss = validate(
        model,
        test_dataloader,
        device,
        use_amp=use_amp,
        pad_token_id=pad_token_id,
    )

    if main_process_flag:
        logger.info("=" * 80)
        logger.info("FINAL TEST SET RESULTS:")
        logger.info(f"  Test Loss: {test_loss:.4f}")
        logger.info(f"  Best Validation Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
        logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--job_id", type=int, help="Job ID")
    parser.add_argument("--config_name", type=str, help="Config file name")
    args = parser.parse_args()

    main(job_id=args.job_id, config_name=args.config_name)
