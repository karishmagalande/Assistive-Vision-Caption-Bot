"""CNN encoder and LSTM decoder (Show and Tell style).

The CNN turns an image into one pooled feature vector. That vector sets the
LSTM's initial state, and the LSTM then generates the caption word by word.
"""

import numpy as np
import keras
from keras import layers

from caption_bot.config import BACKBONES, Config


def build_backbone(cfg: Config, trainable_layers: int = 0) -> keras.Model:
    """Pretrained CNN with global average pooling. Only the top
    `trainable_layers` layers are trainable (0 means fully frozen)."""
    cls_name, size, _ = BACKBONES[cfg.backbone]
    cls = getattr(keras.applications, cls_name)
    backbone = cls(
        include_top=False,
        weights="imagenet" if cfg.pretrained else None,
        input_shape=(size, size, 3),
        pooling="avg",
    )
    backbone.trainable = trainable_layers > 0
    if trainable_layers > 0:
        for layer in backbone.layers[:-trainable_layers]:
            layer.trainable = False
        for layer in backbone.layers:  # keep batch norm statistics frozen
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
    return backbone


def preprocess_fn(cfg: Config):
    module = {"mobilenet_v2": keras.applications.mobilenet_v2,
              "inception_v3": keras.applications.inception_v3}[cfg.backbone]
    return module.preprocess_input


class CaptionDecoder(keras.Model):
    def __init__(self, vocab_size: int, embed_dim: int, units: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.init_h = layers.Dense(units, activation="tanh", name="init_h")
        self.init_c = layers.Dense(units, activation="tanh", name="init_c")
        self.embed = layers.Embedding(vocab_size, embed_dim, mask_zero=True, name="embed")
        self.lstm = layers.LSTM(units, return_sequences=True, return_state=True, name="lstm")
        self.drop = layers.Dropout(dropout)
        self.out = layers.Dense(vocab_size, name="logits")

    def initial_state(self, features):
        return [self.init_h(features), self.init_c(features)]

    def step(self, tokens, states, training=False):
        x = self.drop(self.embed(tokens), training=training)
        seq, h, c = self.lstm(x, initial_state=states, training=training)
        return self.out(self.drop(seq, training=training)), [h, c]

    def call(self, inputs, training=False):
        features, tokens = inputs
        logits, _ = self.step(tokens, self.initial_state(features), training=training)
        return logits


class EndToEndCaptioner(keras.Model):
    """Image in, logits out. Used for fine tuning the CNN together with the decoder."""

    def __init__(self, backbone, decoder, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.decoder = decoder

    def call(self, inputs, training=False):
        images, tokens = inputs
        features = self.backbone(images, training=False)  # batch norm stays in inference mode
        return self.decoder((features, tokens), training=training)


def build_decoder(cfg: Config, vocab_size: int) -> CaptionDecoder:
    decoder = CaptionDecoder(vocab_size, cfg.embed_dim, cfg.units, cfg.dropout)
    decoder((np.zeros((1, cfg.feature_dim), "float32"), np.zeros((1, 2), "int32")))
    return decoder


def compile_for_training(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate, clipnorm=5.0),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )


def greedy_decode(decoder, features, start_id, end_id, max_len) -> list[list[int]]:
    features = np.asarray(features, "float32")
    states = decoder.initial_state(features)
    tokens = np.full((len(features), 1), start_id, "int32")
    out = np.zeros((len(features), max_len), "int32")
    done = np.zeros(len(features), bool)
    for t in range(max_len):
        logits, states = decoder.step(tokens, states)
        nxt = np.argmax(np.asarray(logits)[:, -1, :], axis=-1).astype("int32")
        nxt[done] = end_id
        out[:, t] = nxt
        done |= nxt == end_id
        if done.all():
            break
        tokens = nxt[:, None]
    return out.tolist()


def beam_search(decoder, feature, start_id, end_id, max_len, beam_size=3) -> list[int]:
    """Beam search for a single image. Length normalized log probability."""
    feature = np.asarray(feature, "float32")[None, :]
    h, c = (np.asarray(s) for s in decoder.initial_state(feature))
    beams = [([start_id], 0.0, h, c, False)]
    for _ in range(max_len):
        candidates = []
        for seq, score, h, c, finished in beams:
            if finished:
                candidates.append((seq, score, h, c, True))
                continue
            logits, (nh, nc) = decoder.step(np.array([[seq[-1]]], "int32"), [h, c])
            logp = np.asarray(keras.ops.log_softmax(logits[:, -1, :]))[0]
            for tok in np.argsort(logp)[-beam_size:]:
                candidates.append((seq + [int(tok)], score + float(logp[tok]),
                                   np.asarray(nh), np.asarray(nc), int(tok) == end_id))
        beams = sorted(candidates, key=lambda b: b[1] / len(b[0]), reverse=True)[:beam_size]
        if all(b[4] for b in beams):
            break
    return beams[0][0][1:]
