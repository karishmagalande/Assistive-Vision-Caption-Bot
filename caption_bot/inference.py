"""Loads a trained model directory and captions images."""

import time
from pathlib import Path

import numpy as np
from PIL import Image

from caption_bot.config import Config
from caption_bot.model import beam_search, build_backbone, build_decoder, greedy_decode, preprocess_fn
from caption_bot.vocab import Vocabulary


class Captioner:
    def __init__(self, model_dir: str | Path, beam_size: int = 3):
        model_dir = Path(model_dir)
        self.cfg = Config.load(model_dir / "config.json")
        self.vocab = Vocabulary.load(model_dir / "vocab.json")
        self.beam_size = beam_size

        encoder_weights = model_dir / "encoder.weights.h5"
        if encoder_weights.exists():
            # Fine tuned CNN: build without downloading ImageNet weights, then load ours.
            self.cfg.pretrained = False
            self.backbone = build_backbone(self.cfg)
            self.backbone.load_weights(encoder_weights)
        else:
            self.backbone = build_backbone(self.cfg)
        self.decoder = build_decoder(self.cfg, len(self.vocab))
        self.decoder.load_weights(model_dir / "decoder.weights.h5")
        self.preprocess = preprocess_fn(self.cfg)
        self.caption(Image.new("RGB", (64, 64)))  # warm up so the first request is fast

    def _to_array(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize((self.cfg.image_size, self.cfg.image_size))
        return self.preprocess(np.asarray(image, "float32")[None, ...])

    def features(self, image: Image.Image) -> np.ndarray:
        return np.asarray(self.backbone(self._to_array(image), training=False))[0]

    def caption(self, image: Image.Image) -> dict:
        start = time.perf_counter()
        feature = self.features(image)
        if self.beam_size > 1:
            ids = beam_search(self.decoder, feature, self.vocab.start_id, self.vocab.end_id,
                              self.cfg.max_len, self.beam_size)
        else:
            ids = greedy_decode(self.decoder, feature[None, :], self.vocab.start_id,
                                self.vocab.end_id, self.cfg.max_len)[0]
        text = self.vocab.decode(ids)
        text = (text[:1].upper() + text[1:] + ".") if text else "I could not describe this image."
        return {"caption": text, "latency_ms": round((time.perf_counter() - start) * 1000)}
