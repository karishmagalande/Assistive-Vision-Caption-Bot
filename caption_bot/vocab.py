"""Word level vocabulary with start, end, pad, and unknown tokens."""

import json
import re
from collections import Counter
from pathlib import Path

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"
SPECIALS = [PAD, START, END, UNK]


def clean_caption(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z ]", " ", text)
    return [w for w in text.split() if len(w) > 1 or w in {"a"}]


class Vocabulary:
    def __init__(self, words: list[str]):
        self.itos = SPECIALS + [w for w in words if w not in SPECIALS]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def start_id(self) -> int:
        return self.stoi[START]

    @property
    def end_id(self) -> int:
        return self.stoi[END]

    @classmethod
    def build(cls, captions: list[str], min_count: int = 5) -> "Vocabulary":
        counts = Counter(w for c in captions for w in clean_caption(c))
        words = sorted(w for w, n in counts.items() if n >= min_count)
        return cls(words)

    def encode(self, caption: str, max_len: int) -> list[int]:
        """Returns [<start>, w1, ..., <end>] padded to max_len + 2."""
        ids = [self.stoi.get(w, self.stoi[UNK]) for w in clean_caption(caption)][:max_len]
        ids = [self.start_id] + ids + [self.end_id]
        return ids + [self.pad_id] * (max_len + 2 - len(ids))

    def decode(self, ids) -> str:
        words = []
        for i in ids:
            w = self.itos[int(i)]
            if w == END:
                break
            if w not in SPECIALS:
                words.append(w)
        return " ".join(words)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.itos))

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        itos = json.loads(Path(path).read_text())
        return cls(itos[len(SPECIALS):])
