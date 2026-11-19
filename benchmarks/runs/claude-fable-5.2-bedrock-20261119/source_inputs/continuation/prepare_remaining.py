"""Freeze rows 11–100 from the verified 100-row prepared benchmark suite.

Run from the repository root with PYTHONPATH=.
"""
import json
from pathlib import Path

from benchmarks.src.artifacts import digest, encoded, sample_record, utc_now, write_json
from benchmarks.src.evaluation import load_prepared

base = Path(__file__).resolve().parent
config = json.loads((base / "config.json").read_text())
source = Path("benchmarks/.cache/prepared-datasets-100")
output = Path("benchmarks/.cache/prepared-datasets-90-after-10")
manifest = json.loads((source / "manifest.json").read_text())
if output.exists():
    frozen = json.loads((output / "manifest.json").read_text())
    assert frozen["parent_manifest_sha256"] == digest((source / "manifest.json").read_bytes())
    loaded = load_prepared(output, frozen, config["tasks"])
    for name, dataset in loaded.items():
        assert [sample_record(row, i + 10) for i, row in enumerate(dataset)] == manifest["datasets"][name]["samples"][10:]
    print("Existing remaining subsets verified; nothing overwritten")
else:
    loaded = load_prepared(source, manifest, config["tasks"])
    frozen = {"created_at": utc_now(), "selection": "rows 10 through 99 of frozen 100-sample datasets; no shuffle",
              "samples_per_task": 90, "parent_manifest_sha256": digest((source / "manifest.json").read_bytes()),
              "parent_start_index": 10, "datasets": {}}
    for name, dataset in loaded.items():
        subset = dataset.select(range(10, 100))
        assert [sample_record(row, i + 10) for i, row in enumerate(subset)] == manifest["datasets"][name]["samples"][10:]
        identity = {"features": subset.features.to_dict(), "samples": [sample_record(row, i) for i, row in enumerate(subset)]}
        subset.save_to_disk(str(output / name))
        frozen["datasets"][name] = {**manifest["datasets"][name], **identity,
                                    "fingerprint_sha256": digest(encoded(identity).encode()), "preparation_seconds": 0}
    write_json(output / "manifest.json", frozen)
    print("Prepared verified remaining subsets")
