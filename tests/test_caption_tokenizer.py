"""Offline version, artifact, alignment, and preflight checks for caption PTB."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
from types import MappingProxyType, SimpleNamespace

import pytest

from ml_pipes.vlm import Describe
from ml_pipes.vlm.validation import DescribeMetrics
from ml_pipes.vlm.validation import _caption_tokenizer as ptb


@pytest.fixture
def java(monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        if args[-1] == "-version":
            return SimpleNamespace(stdout="", stderr='openjdk version "test"')
        lines = kwargs["input"].split("\n")[:-1]
        return SimpleNamespace(stdout="\n".join(line.lower() for line in lines) + "\n", stderr="")

    monkeypatch.setattr(ptb.shutil, "which", lambda command: "/test/java")
    monkeypatch.setattr(ptb.subprocess, "run", run)
    return calls


@pytest.fixture
def artifacts(monkeypatch, hf_downloads):
    content = {version: f"artifact {version}".encode() for version in ptb.TOKENIZER_ARTIFACTS}
    registry = MappingProxyType({version: ptb._Artifact(f"https://example.test/{version}.jar", hashlib.sha256(data).hexdigest())
                                 for version, data in content.items()})
    monkeypatch.setattr(ptb, "TOKENIZER_ARTIFACTS", registry)
    hf_downloads[0].update({registry[version].url: data for version, data in content.items()})
    return content, hf_downloads[1]


def test_artifact_allowlist_is_immutable_and_default_is_reference_build():
    assert ptb.DEFAULT_TOKENIZER_VERSION == "3.4.1"
    assert set(ptb.TOKENIZER_ARTIFACTS) == {"3.4.1", "4.6.0"}
    assert ptb.TOKENIZER_ARTIFACTS["3.4.1"].sha256 == "2fcb91bb7a111f93d71e264f4ee0e3afd19ba0dde6d21b38605088df9e940399"
    assert ptb.TOKENIZER_ARTIFACTS["4.6.0"].sha256 == "277e42b16199e22e023d6b53335b2d633cecb740e0ebc9bedb9d36baf6db5431"
    with pytest.raises(TypeError):
        ptb.TOKENIZER_ARTIFACTS["new"] = None
    with pytest.raises(FrozenInstanceError):
        ptb.TOKENIZER_ARTIFACTS["3.4.1"].url = "https://example.test/changed.jar"


@pytest.mark.parametrize("version", ["3.4.1", "4.6.0"])
def test_preparation_checks_hash_repairs_corruption_and_reports_provenance(tmp_path, java, artifacts, version):
    tokenizer = ptb.CaptionTokenizer(version, cache_dir=tmp_path).prepare()
    assert len(artifacts[1]) == 1
    assert tokenizer.jar.read_bytes() == artifacts[0][version]
    assert tokenizer.metadata["tokenizer_version"] == version
    assert tokenizer.metadata["tokenizer_sha256"] == hashlib.sha256(artifacts[0][version]).hexdigest()
    assert tokenizer.metadata["java_version"] == 'openjdk version "test"'
    assert "CR/LF to spaces" in tokenizer.metadata["preprocessing"]
    ptb.CaptionTokenizer(version, cache_dir=tmp_path).prepare()
    assert len(artifacts[1]) == 1
    tokenizer.jar.write_bytes(b"corrupt")
    ptb.CaptionTokenizer(version, cache_dir=tmp_path).prepare()
    assert len(artifacts[1]) == 2
    assert tokenizer.jar.read_bytes() == artifacts[0][version]


def test_versions_have_separate_cache_entries(tmp_path, java, artifacts):
    first = ptb.CaptionTokenizer("3.4.1", cache_dir=tmp_path).prepare()
    second = ptb.CaptionTokenizer("4.6.0", cache_dir=tmp_path).prepare()
    assert first.jar != second.jar and len(artifacts[1]) == 2


def test_ptb_flags_encoding_punctuation_filter_empty_lines_and_line_endings(tmp_path, java, artifacts, monkeypatch):
    tokenizer = ptb.CaptionTokenizer(cache_dir=tmp_path).prepare()
    fixture = json.loads((Path(__file__).parent / "fixtures/coco_caption_scores.json").read_text())["ptb_3.4.1"]
    raw = fixture["raw"]
    # This output comes from the independently executed Java reference wrapper.
    outputs = ["a cat ca n't jump .", "", "the dog -- runs ...", "`` café '' -lrb- two cats -rrb- , 1,000.5 !",
               "first line second line", "a blue-red bus ; is n't it ?", "東京の猫", "... !!!"]

    def run(args, **kwargs):
        assert "-preserveLines" in args and "-lowerCase" in args
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["input"] == "\n".join(text.replace("\n", " ") for text in raw) + "\n"
        return SimpleNamespace(stdout="\n".join(outputs) + "\n")

    monkeypatch.setattr(ptb.subprocess, "run", run)
    assert tokenizer.tokenize(raw) == fixture["tokenized"]


def test_embedded_crlf_is_normalized_without_dropping_empty_candidates(tmp_path, java, artifacts):
    tokenizer = ptb.CaptionTokenizer(cache_dir=tmp_path).prepare()
    assert tokenizer.tokenize(["", "A\r\nB\rC\nD", ""]) == ["", "a b c d", ""]
    assert tokenizer.tokenize([]) == []
    assert java[-1][1]["input"] == "\nA B C D\n\n"


def test_alignment_error_propagates(tmp_path, java, artifacts, monkeypatch):
    tokenizer = ptb.CaptionTokenizer(cache_dir=tmp_path).prepare()
    monkeypatch.setattr(ptb.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="missing line\n"))
    with pytest.raises(RuntimeError, match="line alignment"):
        tokenizer.tokenize(["one", "two"])


@pytest.mark.parametrize("version", ["latest", "3.4", "", None, 341, []])
def test_unsupported_versions_fail_without_setup(version, monkeypatch):
    monkeypatch.setattr(ptb.shutil, "which", lambda command: pytest.fail("Must reject before checking Java"))
    with pytest.raises(ValueError, match="Unsupported tokenizer_version"):
        ptb.CaptionTokenizer(version)


def test_checksum_failure_does_not_run_model(tmp_path, java, artifacts, hf_downloads):
    hf_downloads[0].update({artifact.url: b"unexpected build" for artifact in ptb.TOKENIZER_ARTIFACTS.values()})
    backend = lambda request: pytest.fail("Must fail before inference")
    with pytest.raises(ValueError, match="checksum mismatch"):
        Describe(backend).val(dataset=[], cache_dir=tmp_path, max_samples=1, verbose=False)
    assert not list(tmp_path.rglob("*.jar"))


@pytest.mark.parametrize("failure", ["missing", "broken", "incompatible"])
def test_java_setup_failures_stop_before_inference(tmp_path, java, artifacts, monkeypatch, failure):
    if failure == "missing":
        monkeypatch.setattr(ptb.shutil, "which", lambda command: None)
    else:
        original = ptb.subprocess.run

        def run(args, **kwargs):
            if failure == "incompatible" and args[-1] == "-version":
                return original(args, **kwargs)
            raise subprocess.CalledProcessError(1, args, stderr="Java failed")

        monkeypatch.setattr(ptb.subprocess, "run", run)
    with pytest.raises(RuntimeError):
        Describe(lambda request: pytest.fail("Must fail before inference")).val(dataset=[], cache_dir=tmp_path, verbose=False)


def test_final_tokenizer_runtime_error_is_not_a_model_failure(tmp_path, java, artifacts, monkeypatch):
    metrics = DescribeMetrics(cache_dir=tmp_path).prepare()
    metrics.update(["a cat"], None, image_id=1, im_file="a.jpg")

    def run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "java", stderr="runtime error")

    monkeypatch.setattr(ptb.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="runtime error"):
        metrics.finalize()
    assert metrics.failures == []


def test_cli_preflights_java_before_loading_model(tmp_path, java, artifacts, monkeypatch):
    from examples import run_validation

    monkeypatch.setattr(ptb.shutil, "which", lambda command: None)
    monkeypatch.setattr(run_validation, "create_backend", lambda args: pytest.fail("Must not load model"))
    with pytest.raises(SystemExit) as error:
        run_validation.main(["--task", "describe", "--model", "test", "--max-samples", "1"])
    assert error.value.code == 2
