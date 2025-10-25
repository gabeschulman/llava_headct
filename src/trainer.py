import json
import torch
from torch import nn
from src.model import LLaVAHeadCT
from src.dataloader import create_condition_classification_dataloader


def main():
    config = json.load(open("config.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LLaVAHeadCT(
        vision_encoder_weights=config["vision_encoder"]["vision_encoder_weights"],
        projector_input_channels=config["projector"]["input_channels"],
        projector_inner_channels=config["projector"]["inner_channels"],
        projector_out_channels=config["projector"]["out_channels"],
        decoder_model_name=config["decoder"]["model_name"],
    )
    model.to(device)

    dataloader = create_condition_classification_dataloader(
        config["dataset"]["datapath"], config["dataset"]["batch_size"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    num_epochs = config.get("num_epochs", 10)

    for epoch in range(num_epochs):
        total_loss = 0.0
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
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} completed, Average Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "path/to/save/model.pth")


if __name__ == "__main__":
    main()
