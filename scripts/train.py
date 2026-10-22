"""Train the captioning model.

    python -m scripts.train --images data/flickr8k/Images \
        --captions data/flickr8k/captions.txt --out artifacts/model

Stage 1 trains the LSTM decoder on cached CNN features (fast).
Stage 2 (only with --fine-tune-layers N) unfreezes the top N CNN layers and
trains the CNN and decoder together on the images at a low learning rate.
"""

import argparse
import json
from pathlib import Path

import keras

from caption_bot.config import BACKBONES, Config
from caption_bot.data import feature_dataset, image_dataset, load_captions, make_pairs, split_images
from caption_bot.features import load_or_extract
from caption_bot.model import (EndToEndCaptioner, build_backbone, build_decoder,
                               compile_for_training, preprocess_fn)
from caption_bot.vocab import Vocabulary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="folder of image files")
    p.add_argument("--captions", required=True, help="captions.txt or results.csv")
    p.add_argument("--out", default="artifacts/model")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--backbone", default="mobilenet_v2", choices=list(BACKBONES))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--min-word-count", type=int, default=5)
    p.add_argument("--fine-tune-layers", type=int, default=0)
    p.add_argument("--fine-tune-epochs", type=int, default=3)
    p.add_argument("--limit-images", type=int, default=0, help="use only N images (for quick runs)")
    p.add_argument("--no-pretrained", action="store_true", help="random CNN weights (tests only)")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config(backbone=args.backbone, pretrained=not args.no_pretrained, epochs=args.epochs,
                 batch_size=args.batch_size, min_word_count=args.min_word_count,
                 fine_tune_layers=args.fine_tune_layers, fine_tune_epochs=args.fine_tune_epochs)
    keras.utils.set_random_seed(cfg.seed)

    captions = load_captions(args.captions)
    images = [n for n in sorted(captions) if (Path(args.images) / n).exists()]
    if args.limit_images:
        images = images[: args.limit_images]
    print(f"{len(images)} images, {sum(len(captions[n]) for n in images)} captions")

    train_imgs, val_imgs, test_imgs = split_images(images, cfg.seed)
    vocab = Vocabulary.build([c for n in train_imgs for c in captions[n]], cfg.min_word_count)
    print(f"vocabulary: {len(vocab)} words | train {len(train_imgs)} val {len(val_imgs)} test {len(test_imgs)}")

    cfg.save(out / "config.json")
    vocab.save(out / "vocab.json")
    (out / "split.json").write_text(json.dumps(
        {"train": train_imgs, "val": val_imgs, "test": test_imgs}))

    # Stage 1: decoder on cached features
    print("stage 1: extracting CNN features")
    backbone = build_backbone(cfg)
    cache = Path(args.cache_dir) / f"{cfg.backbone}{'' if cfg.pretrained else '_random'}.npz"
    features = load_or_extract(cache, backbone, cfg, args.images, images, cfg.batch_size)

    train_names, train_tok = make_pairs(train_imgs, captions, vocab, cfg.max_len)
    val_names, val_tok = make_pairs(val_imgs, captions, vocab, cfg.max_len)
    train_ds = feature_dataset(train_names, train_tok, features, cfg.batch_size, seed=cfg.seed)
    val_ds = feature_dataset(val_names, val_tok, features, cfg.batch_size, shuffle=False)

    decoder = build_decoder(cfg, len(vocab))
    compile_for_training(decoder, cfg.learning_rate)
    print("stage 1: training decoder")
    hist1 = decoder.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ])
    decoder.save_weights(out / "decoder.weights.h5")
    if not cfg.pretrained:
        # Random CNN weights cannot be recreated at inference time, so keep them.
        backbone.save_weights(out / "encoder.weights.h5")
    history = {"stage1": hist1.history}

    # Stage 2: fine tune the top CNN layers together with the decoder
    if cfg.fine_tune_layers > 0:
        print(f"stage 2: fine tuning top {cfg.fine_tune_layers} CNN layers")
        ft_backbone = build_backbone(cfg, trainable_layers=cfg.fine_tune_layers)
        model = EndToEndCaptioner(ft_backbone, decoder)
        compile_for_training(model, cfg.fine_tune_learning_rate)
        pre = preprocess_fn(cfg)
        ft_train = image_dataset(train_names, train_tok, args.images, cfg.image_size, pre,
                                 cfg.batch_size // 2, seed=cfg.seed)
        ft_val = image_dataset(val_names, val_tok, args.images, cfg.image_size, pre,
                               cfg.batch_size // 2, shuffle=False)
        hist2 = model.fit(ft_train, validation_data=ft_val, epochs=cfg.fine_tune_epochs, callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)])
        ft_backbone.save_weights(out / "encoder.weights.h5")
        decoder.save_weights(out / "decoder.weights.h5")
        history["stage2"] = hist2.history

    (out / "history.json").write_text(json.dumps(
        {k: {m: [float(x) for x in v] for m, v in h.items()} for k, h in history.items()}, indent=2))
    print(f"saved model to {out}")


if __name__ == "__main__":
    main()
