"""Flask web app: upload or take a photo, get a caption, hear it spoken.

    MODEL_DIR=artifacts/model flask --app app.app run
"""

import base64
import io
import os

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}


def create_app(model_dir: str | None = None, captioner=None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB uploads
    state = {"captioner": captioner}
    model_dir = model_dir or os.getenv("MODEL_DIR", "artifacts/model")

    def get_captioner():
        if state["captioner"] is None:
            from caption_bot.inference import Captioner

            state["captioner"] = Captioner(model_dir, beam_size=int(os.getenv("BEAM_SIZE", "3")))
        return state["captioner"]

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.post("/api/caption")
    def caption():
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return jsonify(error="Attach an image in the 'image' field."), 400
        data = upload.read()
        try:
            Image.open(io.BytesIO(data)).verify()
            image = Image.open(io.BytesIO(data))
        except (UnidentifiedImageError, OSError):
            return jsonify(error="That file is not a readable image."), 400
        if image.format not in ALLOWED_FORMATS:
            return jsonify(error=f"Unsupported image type {image.format}."), 400

        result = get_captioner().caption(image)
        if request.args.get("audio") == "1":
            try:
                from caption_bot.speech import synthesize_mp3

                result["audio_mp3_base64"] = base64.b64encode(
                    synthesize_mp3(result["caption"])).decode()
            except Exception as exc:  # speech is optional, the caption still returns
                result["audio_error"] = str(exc)
        return jsonify(result)

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="Image is larger than 8 MB."), 413

    return app


app = create_app()
