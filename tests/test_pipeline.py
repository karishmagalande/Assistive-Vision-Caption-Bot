"""End to end test on a tiny synthetic dataset. No downloads, CPU only.

    python -m pytest -q
"""

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
COLORS = {"red": (220, 30, 30), "blue": (30, 60, 220), "green": (30, 170, 60)}
SHAPES = ["circle", "square"]


def make_dataset(folder: Path) -> Path:
    images = folder / "images"
    images.mkdir(parents=True)
    rows = ["image,caption"]
    i = 0
    for color, rgb in COLORS.items():
        for shape in SHAPES:
            for k in range(5):
                name = f"img_{i:03d}.jpg"
                img = Image.new("RGB", (128, 128), "white")
                d = ImageDraw.Draw(img)
                box = (20 + k * 4, 20 + k * 4, 100 - k * 2, 100 - k * 2)
                (d.ellipse if shape == "circle" else d.rectangle)(box, fill=rgb)
                img.save(images / name)
                rows.append(f"{name},a {color} {shape} on a white background")
                rows.append(f"{name},there is a {color} {shape}")
                i += 1
    captions = folder / "captions.txt"
    captions.write_text("\n".join(rows))
    return captions


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    captions = make_dataset(tmp)
    model_dir = tmp / "model"
    subprocess.run([sys.executable, "-m", "scripts.train", "--images", str(tmp / "images"),
                    "--captions", str(captions), "--out", str(model_dir),
                    "--cache-dir", str(tmp / "cache"), "--epochs", "3", "--batch-size", "8",
                    "--min-word-count", "1", "--no-pretrained"], cwd=ROOT, check=True)
    return tmp, captions, model_dir


def test_training_writes_artifacts(trained):
    _, _, model_dir = trained
    for f in ["config.json", "vocab.json", "split.json", "decoder.weights.h5",
              "encoder.weights.h5", "history.json"]:
        assert (model_dir / f).exists(), f


def test_captioner_returns_text_and_latency(trained):
    from caption_bot.inference import Captioner

    _, _, model_dir = trained
    result = Captioner(model_dir, beam_size=2).caption(Image.new("RGB", (200, 150), "red"))
    assert isinstance(result["caption"], str) and result["caption"]
    assert result["latency_ms"] >= 0


def test_evaluate_writes_bleu(trained):
    tmp, captions, model_dir = trained
    subprocess.run([sys.executable, "-m", "scripts.evaluate", "--images", str(tmp / "images"),
                    "--captions", str(captions), "--model", str(model_dir), "--beam-size", "1"],
                   cwd=ROOT, check=True)
    metrics = json.loads((model_dir / "metrics.json").read_text())
    assert 0 <= metrics["bleu_4"] <= 1 and metrics["test_images"] > 0


def test_flask_api(trained):
    from app.app import create_app

    _, _, model_dir = trained
    client = create_app(str(model_dir)).test_client()
    assert client.get("/health").json == {"status": "ok"}
    assert client.get("/").status_code == 200

    buf = io.BytesIO()
    Image.new("RGB", (120, 90), "blue").save(buf, "JPEG")
    buf.seek(0)
    res = client.post("/api/caption", data={"image": (buf, "photo.jpg")},
                      content_type="multipart/form-data")
    assert res.status_code == 200 and res.json["caption"]

    res = client.post("/api/caption", data={"image": (io.BytesIO(b"not an image"), "x.jpg")},
                      content_type="multipart/form-data")
    assert res.status_code == 400
    assert client.post("/api/caption").status_code == 400


def test_vocab_round_trip():
    from caption_bot.vocab import Vocabulary

    v = Vocabulary.build(["A dog runs on the grass.", "a dog jumps"], min_count=1)
    ids = v.encode("a dog runs", max_len=5)
    assert len(ids) == 7 and v.decode(ids[1:]) == "a dog runs"


def test_loads_flickr30k_pipe_format(tmp_path):
    from caption_bot.data import load_captions

    f = tmp_path / "results.csv"
    f.write_text("image_name| comment_number| comment\n1.jpg| 0| Two dogs play .\n1.jpg| 1| Dogs run .\n")
    assert load_captions(f) == {"1.jpg": ["Two dogs play .", "Dogs run ."]}
