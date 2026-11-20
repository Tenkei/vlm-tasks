"""Preserve declared setup revisions without rewriting historical run records."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from benchmarks.src import task_registry
from benchmarks.src.artifacts import check_run_setup, model_config_revision, revision_fields, save_run_setup
from benchmarks.src.task_registry import TASKS


def test_run_setup_saves_selected_specs_and_preserves_checkpoint_revision(tmp_path):
    config = {"model": "test", "revision": 3, "checkpoint_revision": "hf-checkpoint",
              "tasks": ["classify", "describe"]}
    save_run_setup(tmp_path, config)
    assert json.loads((tmp_path / "config.json").read_text()) == config
    assert {path.stem for path in (tmp_path / "task-specs").glob("*.json")} == {"classify", "describe"}
    for task in config["tasks"]:
        assert json.loads((tmp_path / "task-specs" / f"{task}.json").read_text()) == json.loads(json.dumps(asdict(TASKS[task])))
        assert revision_fields(task, config) == {
            "task_spec_revision": TASKS[task].revision, "model_config_revision": 3,
        }
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    check_run_setup(tmp_path, config)
    assert {path: path.read_bytes() for path in before} == before
    with pytest.raises(FileExistsError):
        save_run_setup(tmp_path, config)


@pytest.mark.parametrize("revision", [None, True, 0, -1, 1.5, "1"])
def test_model_config_revision_requires_a_positive_integer(tmp_path, revision):
    config = {"revision": revision, "tasks": ["classify"]}
    with pytest.raises(ValueError, match="Model config revision must be a positive integer"):
        save_run_setup(tmp_path, config)
    assert not list(tmp_path.iterdir())


def test_legacy_config_does_not_silently_receive_a_revision():
    with pytest.raises(ValueError, match="Model config revision"):
        model_config_revision({"tasks": ["classify"]})


@pytest.mark.parametrize("change", ["model_revision", "model_setting", "task_revision", "task_context", "missing_spec"])
def test_setup_check_rejects_changed_revisions_or_definitions(tmp_path, monkeypatch, change):
    config = {"revision": 1, "tasks": ["describe"], "temperature": 0}
    save_run_setup(tmp_path, config)
    if change == "model_revision":
        config = {**config, "revision": 2}
    elif change == "model_setting":
        config = {**config, "temperature": 1}
    elif change == "missing_spec":
        (tmp_path / "task-specs/describe.json").unlink()
    else:
        settings = {"revision": 2} if change == "task_revision" else {"context": "A changed prompt."}
        monkeypatch.setattr(task_registry, "TASKS", {**TASKS, "describe": replace(TASKS["describe"], **settings)})
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    with pytest.raises(ValueError, match="differs|no saved task spec"):
        check_run_setup(tmp_path, config)
    assert {path: path.read_bytes() for path in before} == before


def test_setup_files_declare_independent_positive_revisions():
    assert all(type(spec.revision) is int and spec.revision > 0 for spec in TASKS.values())
    root = Path(__file__).resolve().parents[1]
    for path in (root / "benchmarks/setup/model-configs").glob("*.json"):
        assert model_config_revision(json.loads(path.read_text())) > 0
