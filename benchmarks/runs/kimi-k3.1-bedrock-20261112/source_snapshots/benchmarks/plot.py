"""Render reproducible radar charts from saved reports without model calls."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from .common import digest, utc_now, write_json
from .summarize import format_number, load_reports, summarize

# Every selected metric has a natural [0, 1] range. Never scale to observed models.
AXES = [
    ("classify", "accuracy_top1", "Classification", "Accuracy"),
    ("detect", "f1", "Detection", "F1@0.5"),
    ("ground_box", "accuracy_iou_0.5", "Grounding (Box)", "Acc@0.5"),
    ("ground_point", "point_in_box_accuracy", "Grounding (Point)", "Accuracy"),
    ("attributes", "accuracy", "Attributes", "Accuracy"),
    ("answer", "vqa_accuracy", "VQA", "Accuracy"),
    ("describe", "rouge_l", "Captioning", "ROUGE-L"),
]
COLORS = ["#2563ad", "#c05822", "#138575", "#864b9a"]


def plot(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    directory = Path(directory)
    reports = load_reports(directory)
    grouped = {}
    for path, report in reports:
        grouped.setdefault(path.parent.name, {})[report["task"]] = report

    profiles = []
    for run, tasks in sorted(grouped.items(), key=lambda item: next(iter(item[1].values()))["model"]):
        missing = [task for task, _, _, _ in AXES if task not in tasks]
        if missing:
            raise ValueError(f"{run}: missing chart tasks: {', '.join(missing)}")
        scores = [float(tasks[task]["metrics"][metric]) for task, metric, _, _ in AXES]
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in scores):
            raise ValueError(f"{run}: chart metrics must be finite and in [0, 1]")
        first = tasks[AXES[0][0]]
        profiles.append({
            "run": run, "model": first["model"], "backend": first.get("backend", "transformers"),
            "scores": scores,
            "samples": {task: tasks[task]["metrics"]["samples"] for task, *_ in AXES},
        })
    if not profiles:
        raise ValueError("No task reports found")

    output = directory / "charts"
    output.mkdir(parents=True, exist_ok=True)
    angles = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False)
    closed_angles = np.append(angles, angles[0])
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "svg.fonttype": "none", "svg.hashsalt": "vlm-task-profiles",
        "figure.facecolor": "#ffffff", "savefig.facecolor": "#ffffff",
    })

    def draw(ax, profile, index):
        color = COLORS[index % len(COLORS)]
        values = np.array(profile["scores"]) * 100
        closed = np.append(values, values[0])
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels([])
        ax.set_facecolor("#f8fafc")
        ax.grid(color="#d3dbe4", linewidth=.8)
        ax.spines["polar"].set_color("#c4cdd8")
        labels = [f"{label}\n{metric} {format_number(value, 1)}"
                  for (_, _, label, metric), value in zip(AXES, values)]
        ax.set_thetagrids(np.degrees(angles), labels)
        ax.tick_params(axis="x", pad=17, labelsize=10)
        for label, angle in zip(ax.get_xticklabels(), angles):
            label.set_horizontalalignment(
                "center" if abs(np.sin(angle)) < .01 else "left" if np.sin(angle) > 0 else "right"
            )
            label.set_color("#243244")
        ax.plot(closed_angles, closed, color=color, linewidth=2.5, zorder=3)
        ax.fill(closed_angles, closed, color=color, alpha=.15, zorder=2)
        ax.scatter(angles, values, color=color, s=30, edgecolor="white", linewidth=.8, zorder=4)

    def save(profile, index):
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
        fig.subplots_adjust(left=.23, right=.77, bottom=.15, top=.77)
        draw(ax, profile, index)
        title = profile["model"].removeprefix("Qwen/").removeprefix("qwen.")
        fig.suptitle(title, y=.96, fontsize=21, fontweight="bold", color=COLORS[index % len(COLORS)])
        counts = sorted(set(profile["samples"].values()))
        count_text = f"N = {counts[0]}/task" if len(counts) == 1 else "N varies by task"
        fig.text(.5, .91, count_text, ha="center", fontsize=12, color="#596778")
        fig.text(.5, .055, "Higher is better · Failures included",
                 ha="center", fontsize=11, color="#596778")
        fig.savefig(output / f"{profile['run']}.png", dpi=180)
        fig.savefig(output / f"{profile['run']}.svg", metadata={"Date": None})
        plt.close(fig)

    for index, profile in enumerate(profiles):
        save(profile, index)
    for suffix in ("png", "svg"):
        (output / f"task-profiles.{suffix}").unlink(missing_ok=True)
    write_json(output / "manifest.json", {
        "generated_at": utc_now(), "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__, "radial_range": [0, 100], "score_multiplier": 100,
        "axes": [{"task": task, "metric": metric, "label": label, "metric_label": metric_label}
                 for task, metric, label, metric_label in AXES],
        "models": profiles,
        "source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])): digest(path.read_bytes())
                          for path in (Path(__file__).resolve(), Path(__file__).with_name("summarize.py").resolve())},
        "reports_sha256": {str(path.relative_to(directory)): digest(path.read_bytes()) for path, _ in reports},
        "figures_sha256": {path.name: digest(path.read_bytes())
                           for profile in profiles for suffix in ("png", "svg")
                           for path in [output / f"{profile['run']}.{suffix}"]},
    })
    summarize(directory)
    return profiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results"))
    plot(parser.parse_args().results)
