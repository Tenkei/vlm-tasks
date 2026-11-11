"""Isolated HF caches and deterministic offline source downloads."""
from __future__ import annotations

import hashlib
from pathlib import Path

import datasets
import pytest


@pytest.fixture(autouse=True)
def hf_cache(tmp_path, monkeypatch):
    root = tmp_path / 'hf-cache'
    monkeypatch.setattr(datasets.config, 'HF_DATASETS_CACHE', root)
    monkeypatch.setattr(datasets.config, 'DOWNLOADED_DATASETS_PATH', root / 'downloads')
    return root


@pytest.fixture
def hf_downloads(tmp_path, monkeypatch):
    from datasets.download import download_manager

    sources, calls = {}, []
    original = download_manager.cached_path

    def cached_path(url, download_config=None, **kwargs):
        url = str(url)
        if not url.startswith(('http://', 'https://')):
            return original(url, download_config=download_config, **kwargs)
        if url not in sources:
            raise AssertionError(f'Unexpected download: {url}')
        path = Path(download_config.cache_dir or tmp_path / 'downloads') / hashlib.sha256(url.encode()).hexdigest()
        if not path.is_file() or download_config.force_download:
            calls.append(url)
            content = sources[url]
            if isinstance(content, Exception):
                raise content
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        return str(path)

    monkeypatch.setattr(download_manager, 'cached_path', cached_path)
    return sources, calls
