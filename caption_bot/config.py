"""Model and training configuration, saved next to the trained weights."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

BACKBONES = {
    # name: (keras.applications class name, input size, pooled feature size)
    "mobilenet_v2": ("MobileNetV2", 224, 1280),
    "inception_v3": ("InceptionV3", 299, 2048),
}


@dataclass
class Config:
    backbone: str = "mobilenet_v2"
    pretrained: bool = True
    embed_dim: int = 256
    units: int = 512
    dropout: float = 0.3
    max_len: int = 30
    min_word_count: int = 5
    batch_size: int = 64
    epochs: int = 15
    learning_rate: float = 1e-3
    fine_tune_layers: int = 0
    fine_tune_epochs: int = 3
    fine_tune_learning_rate: float = 1e-5
    seed: int = 42
    extra: dict = field(default_factory=dict)

    @property
    def image_size(self) -> int:
        return BACKBONES[self.backbone][1]

    @property
    def feature_dim(self) -> int:
        return BACKBONES[self.backbone][2]

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Config":
        return cls(**json.loads(Path(path).read_text()))
