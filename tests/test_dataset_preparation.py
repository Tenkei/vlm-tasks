"""HF source acquisition, canonical datasets, and default validation loading."""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import tarfile
from zipfile import ZipFile

import cv2
import datasets as hf
import numpy as np
from PIL import Image
import pytest

from ml_pipes.vlm import Classify, Detect, Ground, ModelResponse
from ml_pipes.vlm import validation
from ml_pipes.vlm.validation import ClassificationValidator, load_classification, load_refcocog, load_cifar10, load_coco128
from ml_pipes.vlm.validation import default_datasets as constants


@pytest.fixture
def http(hf_downloads):
    return hf_downloads


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
    sources[constants.COCO_URL] = content
    monkeypatch.setattr(constants, "COCO_CHECKSUM", "sha256:" + hashlib.sha256(content).hexdigest())


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
    sources[constants.MDETR_URL] = tar_bytes(members)
    for filename in set(filenames):
        sources[constants.COCO_IMAGE_URL + filename] = jpeg()
    return data


@pytest.fixture
def cifar(monkeypatch):
    calls = []
    original = hf.load_dataset

    def load(path, **kwargs):
        if path != constants.CIFAR_REPO:
            return original(path, **kwargs)
        calls.append(kwargs)
        return hf.Dataset.from_dict({
            'img': [Image.new('RGB', (32, 32), (180, 90, 30))] * 4,
            'label': [5, 3, 3, 5],
        }, features=hf.Features({'img': hf.Image(), 'label': hf.ClassLabel(names=list(constants.CIFAR_NAMES))}))

    monkeypatch.setattr(hf, 'load_dataset', load)
    return calls


def test_cifar_uses_pinned_hub_revision_and_filters_before_limit(tmp_path, cifar):
    dataset = load_cifar10(split='val', classes=['cat'], max_samples=1, cache_dir=tmp_path)
    assert len(dataset) == 1
    assert dataset.features['label'].names == list(constants.CIFAR_NAMES)
    assert dataset[0]['label'] == 3
    np.testing.assert_array_equal(dataset[0]['image'][0, 0], [180, 90, 30])
    assert cifar == [{'revision': constants.CIFAR_REVISION, 'split': 'test', 'cache_dir': tmp_path}]


@pytest.mark.parametrize('split', ['train', 'test'])
def test_cifar_passes_requested_split_to_hf(split, cifar):
    assert len(load_cifar10(split=split, max_samples=2)) == 2
    assert cifar[0]['split'] == split


def test_coco_returns_native_features_and_reuses_completed_arrow(tmp_path, http, monkeypatch):
    coco_source(http, monkeypatch)
    first = load_coco128(max_samples=2, cache_dir=tmp_path)
    assert isinstance(first, hf.Dataset)
    assert first[0]['objects']['bbox'] == [[30, 15, 90, 45]]
    assert first.features['objects']['category'].feature.names == list(constants.COCO_NAMES)
    assert first.info.dataset_name == 'coco128'
    assert first.info.splits['val'].num_examples == 2
    assert not (tmp_path / 'outside').exists()
    assert not list(tmp_path.rglob('prepared.json'))
    count = len(http[1])
    monkeypatch.setattr(hf.DownloadManager, 'download', lambda *args, **kwargs: pytest.fail('Completed Arrow cache must be reused'))
    again = load_coco128(max_samples=2, cache_dir=tmp_path)
    assert again._fingerprint == first._fingerprint and len(http[1]) == count


@pytest.mark.parametrize('split', ['train', 'val', 'test'])
def test_refcocog_preserves_expressions_and_downloads_distinct_selected_images(tmp_path, http, split):
    ground_source(http)
    dataset = load_refcocog(split=split, max_samples=2, cache_dir=tmp_path)
    assert [row['sample_id'] for row in dataset] == [91, 7]
    assert dataset.features['label'].names == ['cat']
    assert dataset[0]['bbox'] == [30, 15, 90, 45]
    assert len(http[1]) == 2
    assert str(dataset.split) == split
    assert 'UMD' in dataset.info.description
    assert not list(tmp_path.rglob('finetune_*.json'))
    count = len(http[1])
    again = load_refcocog(split=split, max_samples=2, cache_dir=tmp_path)
    assert len(again) == 2 and len(http[1]) == count


def test_larger_grounding_request_reuses_source_cache_with_distinct_arrow_config(tmp_path, http):
    ground_source(http)
    first = load_refcocog(max_samples=1, cache_dir=tmp_path)
    second = load_refcocog(max_samples=3, cache_dir=tmp_path)
    assert len(first) == 1 and len(second) == 3
    assert first.cache_files != second.cache_files
    assert http[1].count(constants.MDETR_URL) == 1 and len(http[1]) == 3


def test_exported_grounding_dataset_keeps_images_when_source_cache_is_removed(tmp_path, http):
    ground_source(http)
    dataset = load_refcocog(max_samples=1, cache_dir=tmp_path)
    expected = dataset[0]['image'].copy()
    destination = tmp_path / 'export'
    dataset.save_to_disk(destination)
    Path(dataset[0]['im_file']).unlink()
    restored = hf.load_from_disk(destination)
    assert restored.features['label'].names == ['cat']
    np.testing.assert_array_equal(restored[0]['image'], expected)
    metrics = Ground(lambda request: ModelResponse({'status': 'not_found', 'box': None})).val(dataset=restored, verbose=False)
    assert metrics.seen == 1 and metrics.dataset_metadata['dataset_name'] == 'refcocog'


def test_invalid_archive_checksum_does_not_publish_arrow_dataset(tmp_path, http):
    http[0][constants.COCO_URL] = b'wrong archive'
    with pytest.raises(ValueError, match='checksum mismatch'):
        load_coco128(cache_dir=tmp_path)
    assert not list(tmp_path.rglob('*.arrow'))


@pytest.mark.parametrize('loader, settings', [
    (load_coco128, {'split': 'test'}),
    (load_refcocog, {'split': '../val'}),
    (load_cifar10, {'max_samples': 0}),
    (load_cifar10, {'max_samples': True}),
    (load_cifar10, {'classes': ['unknown']}),
    (validation.load_ovad, {'groups': ['color']}),
    (validation.load_vqav2, {'split': 'test'}),
    (validation.load_coco_captions2017, {'split': 'test'}),
])
def test_invalid_requests_fail_before_download(tmp_path, http, loader, settings):
    with pytest.raises((ValueError, TypeError)):
        loader(cache_dir=tmp_path, **settings)
    assert not http[1]


@pytest.mark.parametrize('geometry', ['box', 'point'])
def test_ground_val_loads_default_hf_dataset_and_reports_evaluated_counts(tmp_path, http, geometry):
    ground_source(http)
    response = {'status': 'found', 'point': {'x': .5, 'y': .5}} if geometry == 'point' else {
        'status': 'found', 'box': {'x1': .25, 'y1': .25, 'x2': .75, 'y2': .75},
    }
    metrics = Ground(lambda request: ModelResponse(response), geometry=geometry).val(max_samples=2, cache_dir=tmp_path, verbose=False)
    assert metrics.seen == 2 and metrics.unique_images == 1
    assert metrics.dataset_metadata['dataset_name'] == 'refcocog'
    assert metrics.dataset_counts == {'evaluated_samples': 2, 'evaluated_expressions': 2, 'evaluated_images': 1}
    score = 'point_in_box_accuracy' if geometry == 'point' else 'mean_iou'
    assert metrics.results_dict[score] == 1


def test_classification_default_uses_complete_hub_vocabulary(tmp_path, cifar):
    task = Classify(lambda request: ModelResponse({'labels': ['cat']}), ['feline'])
    metrics = task.val(max_samples=2, cache_dir=tmp_path, verbose=False)
    assert metrics.top1 == .5 and metrics.names == tuple(constants.CIFAR_NAMES)
    assert task.labels == {'feline': ''}


def test_task_val_loads_dataset_before_constructing_validator(tmp_path, cifar, monkeypatch):
    events = []
    load = validation.load_cifar10

    def record_loading(**kwargs):
        events.append('load')
        return load(**kwargs)

    class RecordingValidator(ClassificationValidator):
        def __init__(self, task, dataset, **kwargs):
            events.append('construct')
            assert isinstance(dataset, hf.Dataset) and dataset.features['label'].names
            super().__init__(task, dataset, **kwargs)

        def setup(self):
            events.append('setup')

    monkeypatch.setattr(validation, 'load_cifar10', record_loading)
    task = Classify(lambda request: ModelResponse({'labels': ['dog']}))
    assert task.val(validator=RecordingValidator, max_samples=1, cache_dir=tmp_path, verbose=False).top1 == 1
    assert events == ['load', 'construct', 'setup']


def test_supplied_dataset_skips_default_loading_even_when_setup_fails(tmp_path, http):
    class UnavailableValidator(ClassificationValidator):
        def setup(self):
            raise RuntimeError('metric unavailable')

    task = Classify(lambda request: pytest.fail('Must not run inference'))
    with pytest.raises(RuntimeError, match='metric unavailable'):
        task.val(dataset=[], validator=UnavailableValidator, cache_dir=tmp_path)
    assert not http[1]


def test_direct_validators_require_loaded_inputs(tmp_path, http):
    with pytest.raises(TypeError, match='loaded dataset'):
        ClassificationValidator(Classify(lambda request: None), 'cifar10', cache_dir=tmp_path, verbose=False)
    assert not http[1]


def test_runner_grounding_default_builds_hf_dataset_and_saves_report(tmp_path, http, monkeypatch, capsys):
    from examples import run_validation

    ground_source(http)
    response = {'status': 'found', 'box': {'x1': .25, 'y1': .25, 'x2': .75, 'y2': .75}}
    monkeypatch.setattr(run_validation, 'create_backend', lambda args: lambda request: ModelResponse(response))
    output = tmp_path / 'report.json'
    assert run_validation.main(['--task', 'ground', '--model', 'test', '--max-samples', '2', '--cache-dir', str(tmp_path), '--output', str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads(output.read_text())
    assert report['metrics']['mean_iou'] == 1
    assert report['dataset_metadata']['dataset_name'] == 'refcocog'
    assert report['dataset_counts']['evaluated_expressions'] == 2
