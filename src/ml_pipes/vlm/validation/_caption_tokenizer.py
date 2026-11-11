"""Versioned Stanford PTB tokenization with the COCO punctuation filter."""
from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess
from types import MappingProxyType

from .prepare_data import _cache_root, _download


@dataclass(frozen=True)
class _Artifact:
    url: str
    sha256: str


DEFAULT_TOKENIZER_VERSION = "3.4.1"
TOKENIZER_ARTIFACTS = MappingProxyType({
    "3.4.1": _Artifact(
        "https://raw.githubusercontent.com/tylin/coco-caption/3a9afb2682141a03e1cdc02b0df6770d2c884f6f/pycocoevalcap/tokenizer/stanford-corenlp-3.4.1.jar",
        "2fcb91bb7a111f93d71e264f4ee0e3afd19ba0dde6d21b38605088df9e940399",
    ),
    "4.6.0": _Artifact(
        "https://repo.maven.apache.org/maven2/edu/stanford/nlp/stanford-corenlp/4.6.0/stanford-corenlp-4.6.0.jar",
        "277e42b16199e22e023d6b53335b2d633cecb740e0ebc9bedb9d36baf6db5431",
    ),
})
_PUNCTUATION = frozenset(("''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-",
                          ".", "?", "!", ",", ":", "-", "--", "...", ";"))


class CaptionTokenizer:
    def __init__(self, version=DEFAULT_TOKENIZER_VERSION, *, cache_dir=None):
        if not isinstance(version, str) or version not in TOKENIZER_ARTIFACTS:
            raise ValueError(f"Unsupported tokenizer_version {version!r}; expected {tuple(TOKENIZER_ARTIFACTS)}")
        self.version = version
        self.cache_dir = cache_dir
        self._ready = False

    def prepare(self):
        if self._ready:
            return self
        self.java = shutil.which("java")
        if self.java is None:
            raise RuntimeError("COCO caption validation requires Java on PATH for Stanford PTB tokenization")
        try:
            result = subprocess.run([self.java, "-version"], capture_output=True, text=True, timeout=30, check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"Could not run Java for caption tokenization: {exc}") from exc
        java_version = (result.stderr or result.stdout).strip()
        artifact = TOKENIZER_ARTIFACTS[self.version]
        self.jar = _cache_root(self.cache_dir) / "tokenizers" / "stanford-corenlp" / self.version / f"stanford-corenlp-{self.version}.jar"
        _download(artifact.url, self.jar, checksum="sha256:" + artifact.sha256)
        self.metadata = {
            "tokenizer": "Stanford PTBTokenizer", "tokenizer_version": self.version,
            "tokenizer_sha256": artifact.sha256, "tokenizer_url": artifact.url,
            "java_version": java_version,
            "preprocessing": "CR/LF to spaces; PTB -preserveLines -lowerCase; COCO punctuation token filter",
            "punctuation_tokens": sorted(_PUNCTUATION), "encoding": "UTF-8",
        }
        # Detect incompatible Java/artifact combinations before any inference.
        self._tokenize([""])
        self._ready = True
        return self

    def _tokenize(self, texts):
        content = "\n".join(text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ") for text in texts) + "\n"
        try:
            result = subprocess.run(
                [self.java, "-Dfile.encoding=UTF-8", "-cp", str(self.jar),
                 "edu.stanford.nlp.process.PTBTokenizer", "-preserveLines", "-lowerCase"],
                input=content, capture_output=True, text=True, encoding="utf-8", check=True,
            )
        except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
            detail = getattr(exc, "stderr", "") or str(exc)
            raise RuntimeError(f"Caption tokenization failed: {detail}") from exc
        lines = result.stdout.split("\n")
        if lines[-1] == "":
            lines.pop()
        if len(lines) != len(texts):
            raise RuntimeError(f"Caption tokenizer line alignment failed: expected {len(texts)}, got {len(lines)}")
        return [" ".join(token for token in line.split() if token not in _PUNCTUATION) for line in lines]

    def tokenize(self, texts):
        self.prepare()
        if not texts:
            return []
        return self._tokenize(texts)
