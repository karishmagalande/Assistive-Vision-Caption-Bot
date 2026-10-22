"""Server side text to speech.

The web app speaks captions in the browser with the Web Speech API, so this
is only needed for the command line or for clients that want an audio file.
It uses gTTS, which calls Google's text to speech service over the internet.
"""

import io


def synthesize_mp3(text: str, lang: str = "en") -> bytes:
    try:
        from gtts import gTTS
    except ImportError as exc:
        raise RuntimeError("Install gTTS for server side speech: pip install gTTS") from exc
    buf = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buf)
    return buf.getvalue()
