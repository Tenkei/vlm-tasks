"""Dataset preparation against tiny archives and mocked HTTP responses."""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import tarfile
from urllib.error import URLError
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

from ml_pipes.vlm import Classify, Detect, Ground, ModelResponse
from ml_pipes.vlm.validation import ClassificationDataset, RefCOCODataset, YOLODataset, prepare_dataset
from ml_pipes.vlm.validation import prepare as preparation


def tar_bytes(members):
    stream = BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, content in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, BytesIO(content))
    return stream.getvalue()


def jpeg(value=0):
    success, pixels = cv2.imencode(".jpg", np.full((60, 120, 3), value, np.uint8))
    assert success
    return pixels.tobytes()


@pytest.fixture
def http(monkeypatch):
    sources, calls = {}, []

    def open_url(url):
        calls.append(url)
        if url not in sources:
            raise AssertionError(f"Unexpected download: {url}")
        value = sources[url]
        if isinstance(value, Exception):
            raise value
        return BytesIO(value)

    monkeypatch.setattr(preparation, "_open_url", open_url)
    monkeypatch.setattr(preparation.time, "sleep", lambda seconds: None)
    return sources, calls


def cifar_source(http, monkeypatch):
    sources, _ = http

    def batch(labels):
        records = np.zeros((len(labels), 3073), np.uint8)
        records[:, 0] = labels
        records[:, 1:1025] = 180  # Red, with distinct channels to catch RGB/BGR swaps.
        records[:, 1025:2049] = 90
        records[:, 2049:] = 30
        return records.tobytes()

    members = {"cifar-10-batches-bin/test_batch.bin": batch([5, 3, 3, 5])}
    members.update({f"cifar-10-batches-bin/data_batch_{i}.bin": batch([0, 3]) for i in range(1, 6)})
    members["../../outside"] = b"must never be extracted"
    content = tar_bytes(members)
    sources[preparation._CIFAR_URL] = content
    monkeypatch.setattr(preparation, "_CIFAR_CHECKSUM", "sha256:" + hashlib.sha256(content).hexdigest())


def coco_source(http, monkeypatch):
    sources, _ = http
    stream = BytesIO()
    with ZipFile(stream, "w") as archive:
        for index in (3, 1, 2):
            archive.writestr(f"coco128/images/train2017/{index:012d}.jpg", jpeg(index))
            archive.writestr(f"coco128/labels/train2017/{index:012d}.txt", "0 0.5 0.5 0.5 0.5\n")
        archive.writestr("../../outside", "must never be extracted")
        archive.writestr("coco128/data.yaml", "download: arbitrary command must never run")
    content = stream.getvalue()
    sources[preparation._COCO_URL] = content
    monkeypatch.setattr(preparation, "_COCO_CHECKSUM", "sha256:" + hashlib.sha256(content).hexdigest())


def ground_source(http):
    sources, _ = http
    filenames = [f"COCO_train2014_{index:012d}.jpg" for index in (1, 1, 2)]
    data = {
        "images": [
            {"id": query, "caption": f"the animal {query}", "file_name": filename, "width": 120, "height": 60}
            for query, filename in zip((91, 7, 440), filenames)
        ],
        "annotations": [
            {"id": query, "image_id": query, "category_id": 17, "bbox": [30, 15, 60, 30]}
            for query in (440, 91, 7)
        ],
        "categories": [{"id": 17, "name": "cat"}],
    }
    members = {f"OpenSource/finetune_refcocog_{split}.json": json.dumps(data).encode() for split in ("train", "val", "test")}
    members["../../outside"] = b"must never be extracted"
    sources[preparation._MDETR_URL] = tar_bytes(members)
    for filename in set(filenames):
        sources[preparation._COCO_IMAGE_URL + filename] = jpeg()
    return data


def test_cifar_selects_only_requested_split_classes_and_deterministic_prefix(tmp_path, http, monkeypatch):
    cifar_source(http, monkeypatch)
    prepared = prepare_dataset("cifar10", split="val", max_samples=1, classes=("cat",), cache_dir=tmp_path)
    dataset = ClassificationDataset(prepared.data)
    assert len(dataset) == 1 and set(dataset.names.values()) == set(preparation._CIFAR_NAMES)
    assert dataset.names[int(dataset[0]["cls"][0])] == "cat"
    pixels = cv2.imread(dataset[0]["im_file"])
    assert tuple(pixels[0, 0]) == (30, 90, 180)
    assert prepared.metadata["source_split"] == "test"
    assert prepared.metadata["source_samples"] == 4
    assert not (prepared.data / "train").exists()
    assert not (tmp_path / "outside").exists()
    count = len(http[1])
    assert prepare_dataset("cifar10", max_samples=1, classes=("cat",), cache_dir=tmp_path) == prepared
    assert len(http[1]) == count


def test_cifar_train_and_test_are_distinct_preparations_reusing_archive(tmp_path, http, monkeypatch):
    cifar_source(http, monkeypatch)
    train = prepare_dataset("cifar10", split="train", max_samples=2, cache_dir=tmp_path)
    test = prepare_dataset("cifar10", split="test", max_samples=2, cache_dir=tmp_path)
    assert train.metadata["source_samples"] == 10
    assert test.metadata["source_samples"] == 4
    assert train.data != test.data
    assert len(list((train.data / "train").rglob("*.png"))) == 2
    assert len(http[1]) == 1


def test_missing_class_directory_is_repaired_without_redownload(tmp_path, http, monkeypatch):
    cifar_source(http, monkeypatch)
    first = prepare_dataset("cifar10", max_samples=1, cache_dir=tmp_path)
    (first.data / "val" / "airplane").rmdir()
    again = prepare_dataset("cifar10", max_samples=1, cache_dir=tmp_path)
    assert (again.data / "val" / "airplane").is_dir()
    assert len(http[1]) == 1


def test_coco_whitelists_members_and_generates_local_yaml(tmp_path, http, monkeypatch):
    coco_source(http, monkeypatch)
    prepared = prepare_dataset("coco128", max_samples=2, cache_dir=tmp_path)
    dataset = YOLODataset(prepared.data)
    assert len(dataset) == 2
    assert dataset[0]["im_file"].endswith("000000000001.jpg")
    assert dataset.names[0] == "person" and dataset.names[79] == "toothbrush"
    assert dataset[0]["bboxes"].tolist() == [[30, 15, 90, 45]]
    assert "download" not in json.loads(prepared.data.read_text())
    assert not (tmp_path / "outside").exists()
    assert prepared.metadata["source_samples"] == 3
    assert len(http[1]) == 1
    assert prepare_dataset("coco128", max_samples=2, cache_dir=tmp_path) == prepared
    assert len(http[1]) == 1


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_grounding_downloads_requested_annotation_and_distinct_selected_images(tmp_path, http, split):
    ground_source(http)
    prepared = prepare_dataset("refcocog", split=split, max_samples=2, cache_dir=tmp_path)
    dataset = RefCOCODataset(prepared.data, prepared.images, split=split)
    assert [sample["sample_id"] for sample in dataset.samples] == [91, 7]
    assert dataset.counts == {"expressions": 2, "unique_images": 1}
    assert prepared.metadata["source_expressions"] == 3
    assert prepared.metadata["source_unique_images"] == 2
    assert prepared.metadata["expected_split_by"] == "umd"
    assert len(list((tmp_path / "refcocog" / "sources").glob("*.json"))) == 1
    assert len(http[1]) == 2
    assert prepare_dataset("refcocog", split=split, max_samples=2, cache_dir=tmp_path) == prepared
    assert len(http[1]) == 2


def test_larger_grounding_request_reuses_annotations_and_images(tmp_path, http):
    ground_source(http)
    first = prepare_dataset("refcocog", max_samples=1, cache_dir=tmp_path)
    second = prepare_dataset("refcocog", max_samples=3, cache_dir=tmp_path)
    assert first.data != second.data and first.images == second.images
    assert http[1].count(preparation._MDETR_URL) == 1
    assert len(http[1]) == 3
    assert second.metadata["selected_unique_images"] == 2


def test_corrupt_cached_image_is_downloaded_again(tmp_path, http):
    ground_source(http)
    first = prepare_dataset("refcocog", max_samples=2, cache_dir=tmp_path)
    path = next(first.images.glob("*.jpg"))
    path.write_bytes(b"corrupt")
    second = prepare_dataset("refcocog", max_samples=2, cache_dir=tmp_path)
    assert len(http[1]) == 3
    assert cv2.imread(str(path)) is not None
    assert second == first


def test_checksum_mismatch_never_publishes_download_or_preparation(tmp_path, http):
    http[0][preparation._COCO_URL] = b"wrong archive"
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_dataset("coco128", cache_dir=tmp_path)
    assert not list(tmp_path.rglob("prepared.json"))
    assert not list(tmp_path.rglob("coco128.zip"))
    assert not [path for path in tmp_path.rglob("*") if path.is_file()]


def test_failed_and_interrupted_downloads_do_not_create_cache_entries(tmp_path, http):
    http[0][preparation._COCO_URL] = URLError("offline")
    with pytest.raises(RuntimeError, match="offline"):
        prepare_dataset("coco128", cache_dir=tmp_path)
    assert len(http[1]) == 3
    assert not list(tmp_path.rglob("prepared.json"))
    assert not [path for path in tmp_path.rglob("*") if path.is_file()]


def test_interrupted_response_discards_partial_bytes_before_retry(tmp_path, monkeypatch):
    calls = []

    class Interrupted(BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise OSError("connection reset")
            return super().read(3)

    def open_url(url):
        calls.append(url)
        return Interrupted(b"partial data") if len(calls) == 1 else BytesIO(b"complete")

    monkeypatch.setattr(preparation, "_open_url", open_url)
    monkeypatch.setattr(preparation.time, "sleep", lambda seconds: None)
    output = tmp_path / "archive"
    preparation._download("https://example.test/archive", output)
    assert output.read_bytes() == b"complete" and len(calls) == 2
    assert list(tmp_path.iterdir()) == [output]


def test_local_directory_named_like_builtin_never_downloads(tmp_path, http, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path("cifar10") / "val" / "cat" / "image.png"
    path.parent.mkdir(parents=True)
    assert cv2.imwrite(str(path), np.zeros((10, 20, 3), np.uint8))
    task = Classify(lambda request: ModelResponse({"labels": ["cat"]}), ["cat"])
    assert task.val(data=Path("cifar10"), verbose=False).top1 == 1
    assert not http[1]


@pytest.mark.parametrize("name, settings", [
    ("unknown", {}), ("refcoco", {}), ("coco128", {"split": "test"}),
    ("refcocog", {"split": "../val"}), ("cifar10", {"max_samples": 0}),
    ("cifar10", {"max_samples": True}), ("cifar10", {"classes": ("unknown",)}),
    ("refcocog", {"classes": ("cat",)}),
])
def test_invalid_requests_fail_before_network(tmp_path, http, name, settings):
    with pytest.raises(ValueError):
        prepare_dataset(name, cache_dir=tmp_path, **settings)
    assert not http[1]


def test_environment_cache_location(tmp_path, http, monkeypatch):
    coco_source(http, monkeypatch)
    monkeypatch.setenv("VLM_TASKS_DATASETS_DIR", str(tmp_path))
    prepared = prepare_dataset("coco128", max_samples=1)
    assert prepared.data.is_relative_to(tmp_path)


@pytest.mark.parametrize("geometry", ["box", "point"])
def test_ground_val_uses_default_dataset_and_records_preparation(tmp_path, http, geometry):
    ground_source(http)
    response = {"status": "found", "point": {"x": .5, "y": .5}} if geometry == "point" else {
        "status": "found", "box": {"x1": .25, "y1": .25, "x2": .75, "y2": .75},
    }
    task = Ground(lambda request: ModelResponse(response), geometry=geometry)
    metrics = task.val(max_samples=2, cache_dir=tmp_path, verbose=False)
    assert metrics.seen == 2 and metrics.unique_images == 1
    assert metrics.dataset_metadata["preparation"]["source_expressions"] == 3
    score = "point_in_box_accuracy" if geometry == "point" else "mean_iou"
    assert metrics.results_dict[score] == 1


def test_classification_default_download_respects_task_class_selection(tmp_path, http, monkeypatch):
    cifar_source(http, monkeypatch)
    task = Classify(lambda request: ModelResponse({"labels": ["feline"]}), ["feline"])
    metrics = task.val(class_map={"cat": "feline"}, max_samples=2, cache_dir=tmp_path, verbose=False)
    assert metrics.seen == 2 and metrics.top1 == 1
    assert metrics.dataset_metadata["preparation"]["classes"] == ["cat"]


def test_detection_default_download_and_task_dataset_mismatch(tmp_path, http, monkeypatch):
    coco_source(http, monkeypatch)
    task = Detect(lambda request: ModelResponse({"detections": []}), ["person"])
    metrics = task.val(max_samples=2, cache_dir=tmp_path, verbose=False)
    assert metrics.seen == 2 and metrics.results_dict["fn"] == 2
    count = len(http[1])
    with pytest.raises(ValueError, match="does not support"):
        task.val(data="cifar10", cache_dir=tmp_path, verbose=False)
    assert len(http[1]) == count


def test_runner_prepare_only_skips_backend_and_records_paths(tmp_path, http, monkeypatch, capsys):
    from examples import run_validation

    ground_source(http)
    monkeypatch.setattr(run_validation, "create_backend", lambda args: pytest.fail("Must not load a model"))
    output = tmp_path / "report.json"
    assert run_validation.main([
        "--task", "ground", "--prepare-only", "--max-samples", "2", "--cache-dir", str(tmp_path),
        "--output", str(output),
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads(output.read_text())
    assert report["dataset"] == "refcocog"
    assert Path(report["data"]).is_dir() and Path(report["images"]).is_dir()
    assert report["metadata"]["selected_expressions"] == 2


def test_runner_classification_default_uses_dataset_labels(tmp_path, http, monkeypatch, capsys):
    from examples import run_validation

    cifar_source(http, monkeypatch)
    monkeypatch.setattr(run_validation, "create_backend", lambda args: lambda request: ModelResponse({"labels": ["cat"]}))
    assert run_validation.main([
        "--task", "classify", "--model", "test", "--max-samples", "2",
        "--cache-dir", str(tmp_path), "--no-verbose",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["metrics"]["samples"] == 2 and report["metrics"]["accuracy_top1"] == 1
    assert report["task_config"]["labels"] == list(preparation._CIFAR_NAMES)
    assert report["dataset_metadata"]["preparation"]["dataset"] == "cifar10"
    assert len(http[1]) == 1
