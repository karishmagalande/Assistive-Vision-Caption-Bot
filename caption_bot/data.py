"""Caption file loading, train/val/test split, and tf.data pipelines."""

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_captions(captions_file: Path) -> dict[str, list[str]]:
    """Reads a captions file into {image_name: [caption, ...]}.

    Works with the Kaggle Flickr8k format (image,caption) and the Kaggle
    Flickr30k format (image_name| comment_number| comment). The first column
    is treated as the image name and the last column as the caption.
    """
    text = Path(captions_file).read_text(encoding="utf-8")
    delimiter = "|" if "|" in text.splitlines()[0] else ","
    reader = csv.reader(text.splitlines(), delimiter=delimiter)
    header = next(reader)
    if not header[0].strip().lower().startswith("image"):
        reader = csv.reader(text.splitlines(), delimiter=delimiter)  # no header
    captions = defaultdict(list)
    for row in reader:
        if len(row) < 2:
            continue
        image, caption = row[0].strip(), row[-1].strip()
        if image and caption:
            captions[image].append(caption)
    return dict(captions)


def split_images(images: list[str], seed: int = 42, val: float = 0.1, test: float = 0.1):
    """Splits by image so no image appears in more than one split."""
    images = sorted(images)
    random.Random(seed).shuffle(images)
    n_test, n_val = int(len(images) * test), int(len(images) * val)
    return images[n_test + n_val:], images[n_test:n_test + n_val], images[:n_test]


def make_pairs(images, captions, vocab, max_len):
    """One row per (image, caption). Returns image names and token arrays."""
    names, tokens = [], []
    for img in images:
        for cap in captions[img]:
            names.append(img)
            tokens.append(vocab.encode(cap, max_len))
    return names, np.array(tokens, dtype="int32")


def load_image(path, image_size: int, preprocess):
    import tensorflow as tf

    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (image_size, image_size))
    return preprocess(img)


def feature_dataset(names, tokens, features: dict, batch_size, shuffle=True, seed=42):
    """((image_feature, input_tokens), target_tokens) batches from cached features."""
    import tensorflow as tf

    feats = np.stack([features[n] for n in names]).astype("float32")
    ds = tf.data.Dataset.from_tensor_slices(((feats, tokens[:, :-1]), tokens[:, 1:]))
    if shuffle:
        ds = ds.shuffle(min(len(names), 20000), seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def image_dataset(names, tokens, image_dir, image_size, preprocess, batch_size,
                  shuffle=True, seed=42):
    """((image, input_tokens), target_tokens) batches decoded from image files."""
    import tensorflow as tf

    paths = [str(Path(image_dir) / n) for n in names]
    ds = tf.data.Dataset.from_tensor_slices((paths, tokens))
    if shuffle:
        ds = ds.shuffle(min(len(names), 20000), seed=seed)
    ds = ds.map(
        lambda p, t: ((load_image(p, image_size, preprocess), t[:-1]), t[1:]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
