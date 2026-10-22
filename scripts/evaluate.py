"""BLEU scores and latency on the held out test images.

    python -m scripts.evaluate --images data/flickr8k/Images \
        --captions data/flickr8k/captions.txt --model artifacts/model
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from PIL import Image

from caption_bot.data import load_captions
from caption_bot.inference import Captioner
from caption_bot.vocab import clean_caption


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--captions", required=True)
    p.add_argument("--model", default="artifacts/model")
    p.add_argument("--beam-size", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="evaluate on the first N test images")
    args = p.parse_args()

    captions = load_captions(args.captions)
    test_imgs = json.loads((Path(args.model) / "split.json").read_text())["test"]
    if args.limit:
        test_imgs = test_imgs[: args.limit]
    captioner = Captioner(args.model, beam_size=args.beam_size)

    references, hypotheses, latencies = [], [], []
    for i, name in enumerate(test_imgs, 1):
        image = Image.open(Path(args.images) / name)
        start = time.perf_counter()
        result = captioner.caption(image)
        latencies.append((time.perf_counter() - start) * 1000)
        hypotheses.append(clean_caption(result["caption"]))
        references.append([clean_caption(c) for c in captions[name]])
        print(f"\r  {i}/{len(test_imgs)}", end="", flush=True)
    print()

    smooth = SmoothingFunction().method1
    weights = {1: (1, 0, 0, 0), 2: (0.5, 0.5, 0, 0), 3: (1 / 3, 1 / 3, 1 / 3, 0), 4: (0.25,) * 4}
    metrics = {f"bleu_{n}": round(corpus_bleu(references, hypotheses, weights=w,
                                              smoothing_function=smooth), 4)
               for n, w in weights.items()}
    metrics.update({
        "test_images": len(test_imgs),
        "beam_size": args.beam_size,
        "latency_ms_mean": round(statistics.mean(latencies)),
        "latency_ms_p95": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]),
    })
    print(json.dumps(metrics, indent=2))
    (Path(args.model) / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
