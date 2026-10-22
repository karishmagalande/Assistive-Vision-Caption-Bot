# Assistive Vision Caption Bot

Describes a photo in a sentence and reads it out loud, so a person who is blind or has low vision can find out what is in front of the camera.

A CNN encoder (MobileNetV2 or InceptionV3, pretrained on ImageNet) turns the image into a feature vector. An LSTM decoder, initialised from that vector, writes the caption one word at a time with beam search. A Flask web app lets you take or upload a photo on a phone and speaks the caption with the browser's text to speech.

```
photo ─▶ CNN encoder ─▶ 1280-d feature ─▶ LSTM decoder ─▶ "A dog runs through the grass." ─▶ speech
         (MobileNetV2,                     (beam search)
          top layers fine tuned)
```

## Results

Fill this in from `artifacts/model/metrics.json` after you train and evaluate.

| Dataset | Train images | Backbone | BLEU-1 | BLEU-4 | Mean latency (CPU) |
|---|---|---|---|---|---|
| Flickr8k | | MobileNetV2 | | | |

## Project layout

```
caption_bot/
  config.py      model and training settings, saved with the weights
  vocab.py       word vocabulary and caption cleaning
  data.py        caption loading, train/val/test split by image, tf.data pipelines
  model.py       CNN encoder, LSTM decoder, greedy and beam search decoding
  features.py    caches CNN features so decoder training is fast
  inference.py   Captioner: load a trained model and caption an image
  speech.py      optional server side text to speech (gTTS)
scripts/
  train.py         two stage training (decoder, then optional CNN fine tuning)
  evaluate.py      BLEU 1 to 4 and latency on the held out test images
  caption_image.py caption one image from the command line
app/
  app.py         Flask app and JSON API
  templates/, static/   accessible single page front end
tests/
  test_pipeline.py  end to end test on a synthetic dataset, no downloads
Dockerfile        production image (gunicorn, CPU TensorFlow)
```

## 1. Set up

Python 3.10 to 3.13.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q              # 6 tests, about 30 seconds on CPU
```

## 2. Get a dataset

**Flickr8k** (8,091 images, 5 captions each). Download from Kaggle:
https://www.kaggle.com/datasets/adityajn105/flickr8k

Unzip so you have:

```
data/flickr8k/Images/*.jpg
data/flickr8k/captions.txt
```

**Flickr30k** (31,783 images) works the same way, and the loader reads its `results.csv` pipe format directly:
https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

## 3. Train

```bash
python -m scripts.train \
  --images data/flickr8k/Images \
  --captions data/flickr8k/captions.txt \
  --out artifacts/model \
  --fine-tune-layers 30
```

- **Stage 1** runs MobileNetV2 once over every image, caches the features in `cache/`, and trains the LSTM decoder on them. On a laptop CPU this takes roughly 20 to 40 minutes for Flickr8k.
- **Stage 2** (`--fine-tune-layers 30`) unfreezes the top 30 CNN layers and trains the CNN and decoder together for 3 epochs at a low learning rate. Use a GPU for this, for example Google Colab (see below). Leave the flag out to skip it.
- For a quick test run, add `--limit-images 500 --epochs 3`.
- `--backbone inception_v3` uses a larger, slower CNN that usually scores a little higher.

The first run downloads the ImageNet weights for the CNN automatically.

### Training on Google Colab (free GPU)

1. In a new Colab notebook choose Runtime > Change runtime type > GPU.
2. Run:

```python
!git clone https://github.com/karishmagalande/Final_Year_Project_Engineering.git
%cd Final_Year_Project_Engineering
!pip install -q nltk gTTS
# Upload kaggle.json (Kaggle > Settings > Create New Token) when prompted
from google.colab import files; files.upload()
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d adityajn105/flickr8k -p data/flickr8k --unzip
!python -m scripts.train --images data/flickr8k/Images --captions data/flickr8k/captions.txt \
    --out artifacts/model --fine-tune-layers 30
!python -m scripts.evaluate --images data/flickr8k/Images --captions data/flickr8k/captions.txt \
    --model artifacts/model
!zip -r model.zip artifacts/model
files.download("model.zip")
```

3. Unzip `model.zip` into your local repo so you have `artifacts/model/`.

## 4. Evaluate

```bash
python -m scripts.evaluate \
  --images data/flickr8k/Images \
  --captions data/flickr8k/captions.txt \
  --model artifacts/model
```

This prints BLEU 1 to 4 on the held out test images (10% of the dataset, never seen in training) plus mean and p95 latency, and saves them to `artifacts/model/metrics.json`. Copy those numbers into the Results table above.

## 5. Try it

```bash
python -m scripts.caption_image path/to/photo.jpg
python -m scripts.caption_image path/to/photo.jpg --speak caption.mp3   # also saves audio

MODEL_DIR=artifacts/model flask --app app.app run
# open http://127.0.0.1:5000 and take or choose a photo
```

On a phone, the "Take or choose a photo" button opens the camera directly. The caption is announced to screen readers and spoken aloud.

### API

```bash
curl -F "image=@photo.jpg" http://127.0.0.1:5000/api/caption
# {"caption": "A dog runs through the grass.", "latency_ms": 480}

curl -F "image=@photo.jpg" "http://127.0.0.1:5000/api/caption?audio=1"
# adds "audio_mp3_base64" with the spoken caption
```

## 6. Deploy

Commit `artifacts/model/` first. The weight files are well under GitHub's 100 MB limit.

### Docker (any host)

```bash
docker build -t caption-bot .
docker run -p 7860:7860 caption-bot
# open http://localhost:7860
```

### Hugging Face Spaces (free)

1. Create a new Space at https://huggingface.co/new-space and choose **Docker** as the SDK.
2. Push this repo to the Space (or link it to your GitHub repo).
3. At the very top of the Space's `README.md`, add:

```yaml
---
title: Assistive Vision Caption Bot
sdk: docker
app_port: 7860
---
```

The Space builds the Dockerfile and gives you a public HTTPS link that works on a phone camera.

### Render

Create a **Web Service** from the GitHub repo, choose **Docker**, and deploy. Render sets `PORT` automatically and the Dockerfile reads it.

## Notes

- Captions are generated by beam search with a beam of 3. Set `BEAM_SIZE=1` for greedy decoding, which is faster but slightly less fluent.
- Uploads are limited to 8 MB, and the server checks that the file really is an image before decoding it.
- Server side speech (`?audio=1` and `--speak`) uses gTTS, which needs internet access. The web page itself speaks through the browser and needs nothing extra.
