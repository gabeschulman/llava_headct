from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, Union
import torch
from torch.utils.data import DataLoader

from src.dataloader import create_head_ct_dataloader
from src.model import LLaVAHeadCT

FILEPATH = Path(__file__).parent.resolve()


@dataclass
class ModelConfig:
    """Class to load and manage configuration settings from a JSON file."""

    config_name: str

    def __post_init__(self):
        self.load_config()
        self.encoder_config = self.load_encoder_config()
        self.projector_config = self.load_projector_config()
        self.decoder_config = self.load_decoder_config()
        self.dataset_config = self.load_dataset_config()
        self.update_dataset_config()
        self.train_config = self.load_train_config()
        self.update_train_config()
        self.dataloader_config = self.load_dataloader_config()
        self.update_dataloader_config()
        self.model_state_dict_path = self.load_model_state_dict_path()

    def load_config(self) -> None:
        """Load configuration from the JSON file."""
        if not self.config_name.endswith(".json"):
            self.config_name += ".json"
        config_path = os.path.join(FILEPATH, "configs", self.config_name)
        with open(config_path, "r") as f:
            self.config: Dict[str, Any] = json.load(f)

    def load_encoder_config(self) -> Dict[str, Any]:
        if "encoder" not in self.config:
            raise KeyError("Encoder configuration not found in config file.")
        encoder_config: Dict[str, Any] = self.config["encoder"]
        required_keys = [
            "vision_encoder_weights",
            "vision_encoder_in_chans",
            "vision_encoder_img_size",
            "vision_encoder_patch_size",
        ]
        mistyped_keys = [
            key for key in required_keys[1:] if not isinstance(encoder_config[key], int)
        ]
        mistyped_keys += (
            [required_keys[0]]
            if not isinstance(encoder_config[required_keys[0]], str)
            else []
        )
        if mistyped_keys:
            raise TypeError(
                f"Encoder config keys have incorrect types: {mistyped_keys}"
            )
        return encoder_config

    def load_projector_config(self) -> Dict[str, Any]:
        if "projector" not in self.config:
            raise KeyError("Projector configuration not found in config file.")
        projector_config: Dict[str, Any] = self.config["projector"]
        required_keys = [
            "projector_input_channels",
            "projector_inner_channels",
            "projector_out_channels",
        ]
        missing_keys = [key for key in required_keys if key not in projector_config]
        if missing_keys:
            raise KeyError(f"Missing required projector config keys: {missing_keys}")
        mistyped_keys = [
            key for key in required_keys if not isinstance(projector_config[key], int)
        ]
        if mistyped_keys:
            raise TypeError(f"Projector config keys must be integers: {mistyped_keys}")
        projector_config["projector_dropout"] = projector_config.get(
            "projector_dropout", 0.1
        )
        return projector_config

    def load_decoder_config(self) -> Dict[str, Any]:
        if "decoder" not in self.config:
            raise KeyError("Decoder configuration not found in config file.")
        decoder_config: Dict[str, Any] = self.config["decoder"]
        required_keys = ["decoder_model_name"]
        mistyped_keys = [
            key for key in required_keys if not isinstance(decoder_config[key], str)
        ]
        if mistyped_keys:
            raise TypeError(f"Decoder config keys must be strings: {mistyped_keys}")
        return decoder_config

    def load_dataset_config(self) -> Dict[str, Any]:
        """Load training configuration from the config file."""
        if "dataset" not in self.config:
            raise KeyError("Dataset configuration not found in config file.")
        dataset_config: Dict[str, Any] = self.config["dataset"]
        required_keys = ["filepath", "filenames"]
        missing_keys = [key for key in required_keys if key not in dataset_config]
        if missing_keys:
            raise KeyError(f"Missing required dataset config keys: {missing_keys}")
        if not isinstance(dataset_config["filepath"], str):
            raise TypeError("Dataset config 'filepath' must be a string.")
        if not isinstance(dataset_config["filenames"], dict):
            raise TypeError("Dataset config 'filenames' must be a dictionary.")
        return dataset_config

    def update_dataset_config(self) -> None:
        self.dataset_config["train_file"] = os.path.join(
            self.dataset_config["filepath"], self.dataset_config["filenames"]["train"]
        )
        self.dataset_config["val_file"] = os.path.join(
            self.dataset_config["filepath"], self.dataset_config["filenames"]["val"]
        )
        self.dataset_config["test_file"] = os.path.join(
            self.dataset_config["filepath"], self.dataset_config["filenames"]["test"]
        )

    def load_train_config(self) -> Dict[str, Any]:
        if "training" not in self.config:
            return {}
        train_config: Dict[str, Any] = self.config["training"]
        return train_config

    def update_train_config(self) -> None:
        """Update the training configuration with new values."""
        default_value_map = {
            "num_epochs": 6,
            "gradient_accumulation_steps": 2,
            "max_epochs": 200,
            "base_lr": 5e-5,
            "weight_decay": 0.04,
            "use_amp": True,
        }
        for key, default_value in default_value_map.items():
            self.train_config.setdefault(key, default_value)

    def load_dataloader_config(self) -> Dict[str, Any]:
        if "dataloader" not in self.config:
            return {}
        dataloader_config: Dict[str, Any] = self.config["dataloader"]
        return dataloader_config

    def update_dataloader_config(self) -> None:
        """Update the dataloader configuration with new values."""
        default_value_map = {
            "batch_size": 16,
            "num_workers": 8,
            "max_text_length": 512,
            "roi": [-1000, 400],
            "window_sizes": [[80, 80], [120, 120], [160, 160]],
            "use_cached_images": True,
        }
        for key, default_value in default_value_map.items():
            self.dataloader_config.setdefault(key, default_value)

    def load_model_state_dict_path(self) -> Union[str, None]:
        config_path: Union[str, None] = self.config.get("model_state_dict_path", None)
        return config_path

    def __str__(self):
        return json.dumps(self.config, indent=4)

    def __repr__(self):
        return self.__str__()


@dataclass
class DataLoaderHandler:
    """Class to manage data loader configuration."""

    objective: str
    model_config: ModelConfig
    rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        self.supported_objectives: dict[str, str] = {
            "condition_classification": "conditions",
            "impression_generation": "impression_deid",
            "narrative_generation": "narrative_deid",
            "qa_training": "qa_answer",
        }
        if self.objective not in self.supported_objectives.keys():
            raise ValueError(f"Unsupported objective: {self.objective}")
        self.objective_column = self.supported_objectives[self.objective]
        self.shared_gen_dataloader_args = {
            "objective_column": self.objective_column,
            "tokenizer_model_name": self.model_config.decoder_config[
                "decoder_model_name"
            ],
            "rank": self.rank,
            "world_size": self.world_size,
        }

    def get_train_dataloader(self) -> DataLoader:
        return create_head_ct_dataloader(
            image_file_location=self.model_config.dataset_config["train_file"],
            **self.shared_gen_dataloader_args,
            **self.model_config.dataloader_config,
        )

    def get_val_dataloader(self) -> DataLoader:
        return create_head_ct_dataloader(
            image_file_location=self.model_config.dataset_config["val_file"],
            **self.shared_gen_dataloader_args,
            **self.model_config.dataloader_config,
        )

    def get_test_dataloader(self) -> DataLoader:
        return create_head_ct_dataloader(
            image_file_location=self.model_config.dataset_config["test_file"],
            **self.shared_gen_dataloader_args,
            **self.model_config.dataloader_config,
        )

    def get_objective_prompt_tokens(
        self,
        model: Union[LLaVAHeadCT, torch.nn.parallel.DistributedDataParallel],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_templates: dict[str, str] = {
            "condition_classification": "Describe the medical conditions observed in the attached head CT scan. Please list the conditions present in the following format: 'Conditions: condition 1, condition 2, ... condition N'. If no abnormalities are observed, please respond with 'Conditions: none.'",
            "impression_generation": "Provide a concise radiologist's medical impression based on the findings from the attached head CT scan.",
            "narrative_generation": "Generate a detailed radiologist's medical narrative based on the findings from the attached head CT scan.",
            "qa_training": "Answer the following question about this CT scan:",
        }
        text_tokens: dict[str, torch.Tensor] = model.decoder.tokenizer(  # type: ignore
            prompt_templates[self.objective],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return text_tokens["input_ids"].to(device), text_tokens["attention_mask"].to(
            device
        )
