"""Runs the CNN once over every image and caches the pooled features."""

from pathlib import Path

import numpy as np

from caption_bot.data import load_image
from caption_bot.model import preprocess_fn


def extract_features(backbone, cfg, image_dir, image_names, batch_size=64) -> dict:
    import tensorflow as tf

    preprocess = preprocess_fn(cfg)
    paths = [str(Path(image_dir) / n) for n in image_names]
    ds = (tf.data.Dataset.from_tensor_slices(paths)
          .map(lambda p: load_image(p, cfg.image_size, preprocess),
               num_parallel_calls=tf.data.AUTOTUNE)
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE))
    vectors = []
    for i, batch in enumerate(ds):
        vectors.append(np.asarray(backbone(batch, training=False)))
        done = min((i + 1) * batch_size, len(paths))
        print(f"\r  features {done}/{len(paths)}", end="", flush=True)
    print()
    vectors = np.concatenate(vectors) if vectors else np.zeros((0, cfg.feature_dim))
    return dict(zip(image_names, vectors))


def load_or_extract(cache_file, backbone, cfg, image_dir, image_names, batch_size=64) -> dict:
    cache_file = Path(cache_file)
    if cache_file.exists():
        cached = dict(np.load(cache_file))
        if all(n in cached for n in image_names):
            print(f"  using cached features from {cache_file}")
            return cached
    features = extract_features(backbone, cfg, image_dir, image_names, batch_size)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, **features)
    return features
