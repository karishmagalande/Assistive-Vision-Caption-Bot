"""Caption one image from the command line, and optionally save spoken audio.

    python -m scripts.caption_image photo.jpg
    python -m scripts.caption_image photo.jpg --speak caption.mp3
"""

import argparse
import json

from PIL import Image

from caption_bot.inference import Captioner
from caption_bot.speech import synthesize_mp3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--model", default="artifacts/model")
    p.add_argument("--beam-size", type=int, default=3)
    p.add_argument("--speak", metavar="OUT.mp3", help="also save the caption as speech")
    args = p.parse_args()

    result = Captioner(args.model, beam_size=args.beam_size).caption(Image.open(args.image))
    print(json.dumps(result, indent=2))
    if args.speak:
        with open(args.speak, "wb") as f:
            f.write(synthesize_mp3(result["caption"]))
        print(f"saved speech to {args.speak}")


if __name__ == "__main__":
    main()
