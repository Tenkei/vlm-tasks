"""Load shared benchmark definitions independently of model configurations."""
import json

import pytest

from benchmarks.src.task_registry import AXES, TASKS, TASK_SPECS_DIRECTORY, load_task_specs


def test_task_specs_preserve_the_benchmark_task_and_axis_order():
    expected = ["classify", "detect", "ground_box", "ground_point", "attributes", "answer", "describe"]
    assert list(TASKS) == [task for task, *_ in AXES] == expected
    assert TASKS["ground_box"].geometry == "box"
    assert TASKS["ground_point"].geometry == "point"
    assert TASKS["describe"].primary_metric == "rouge_l"
    assert "10 to 15 words" in TASKS["describe"].context


def test_task_specs_load_json_values_in_explicit_order(tmp_path):
    first = json.loads((TASK_SPECS_DIRECTORY / "describe.json").read_text())
    first.update(order=1, title="Custom captions", context="Focus on visible text.")
    second = json.loads((TASK_SPECS_DIRECTORY / "ground_point.json").read_text())
    second["order"] = 2
    (tmp_path / "z_first.json").write_text(json.dumps(first))
    (tmp_path / "a_second.json").write_text(json.dumps(second))
    tasks = load_task_specs(tmp_path)
    assert list(tasks) == ["z_first", "a_second"]
    assert tasks["z_first"].title == "Custom captions"
    assert tasks["z_first"].context == "Focus on visible text."
    assert tasks["z_first"].metrics[-1] == ("rouge_l", "ROUGE-L")
    assert tasks["a_second"].geometry == "point"


@pytest.mark.parametrize("change", [
    {"order": 0}, {"order": True}, {"order": "1"},
    {"revision": 0}, {"revision": True},
    {"revision": "1"}, {"revision": None},
    {"primary_metric": "missing"}, {"metrics": [["rouge_l"]]},
])
def test_task_specs_reject_invalid_revision_order_or_metrics(tmp_path, change):
    data = json.loads((TASK_SPECS_DIRECTORY / "describe.json").read_text())
    (tmp_path / "describe.json").write_text(json.dumps({**data, **change}))
    with pytest.raises(ValueError, match="Invalid task spec"):
        load_task_specs(tmp_path)


def test_task_specs_reject_ambiguous_order_and_missing_definitions(tmp_path):
    with pytest.raises(ValueError, match="No task specs"):
        load_task_specs(tmp_path)
    data = (TASK_SPECS_DIRECTORY / "describe.json").read_text()
    (tmp_path / "first.json").write_text(data)
    (tmp_path / "second.json").write_text(data)
    with pytest.raises(ValueError, match="unique positive integer"):
        load_task_specs(tmp_path)
