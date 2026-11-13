"""Render reproducible model profiles and task comparisons from saved reports."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from .artifacts import digest, utc_now, write_json
from .generate_report import format_number, generate_report, load_task_reports
from .task_registry import AXES

COLORS = ["#2563ad", "#c05822", "#138575", "#864b9a"]
MODEL_LABELS = {
    "Qwen/Qwen3-VL-2B-Instruct": "Qwen3-VL-2B",
    "google.gemma-3-27b-it": "Gemma 3 27B",
    "moonshotai.kimi-k3.1": "Kimi K3.1",
    "qwen.qwen3-vl-235b-a22b": "Qwen3-VL-235B",
}


def generate_charts(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    directory = Path(directory)
    reports = load_task_reports(directory)
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
            "scores": scores, "figure": f"model-profiles/{run}",
            "samples": {task: tasks[task]["metrics"]["samples"] for task, *_ in AXES},
        })
    if not profiles:
        raise ValueError("No task reports found")

    output = directory.parent / "reports/charts"
    (output / "model-profiles").mkdir(parents=True, exist_ok=True)
    (output / "task-comparisons").mkdir(parents=True, exist_ok=True)
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
        labels = [r"$\mathbf{" + label.replace(" ", r"\ ") + "}$\n"
                  + f"{metric} {format_number(value, 1)}"
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
        fig.text(.5, .055, "Higher is better · Failures included",
                 ha="center", fontsize=11, color="#596778")
        fig.savefig(output / f"{profile['figure']}.png", dpi=180)
        fig.savefig(output / f"{profile['figure']}.svg", metadata={"Date": None})
        plt.close(fig)

    for index, profile in enumerate(profiles):
        save(profile, index)

    task_comparisons = []
    for axis, (task, metric, title, metric_label) in enumerate(AXES):
        models = [{"run": profile["run"], "model": profile["model"],
                   "label": MODEL_LABELS.get(profile["model"], profile["model"]),
                   "score": profile["scores"][axis]}
                  for profile in profiles]
        figure = f"task-comparisons/{task}"
        fig, ax = plt.subplots(figsize=(8, 4.8))
        fig.subplots_adjust(left=.29, right=.88, bottom=.18, top=.76)
        values = [model["score"] * 100 for model in models]
        positions = np.arange(len(models))
        ax.set_facecolor("#f8fafc")
        ax.set_axisbelow(True)
        ax.barh(positions, values, height=.55,
                color=[COLORS[index % len(COLORS)] for index in range(len(models))])
        ax.set_yticks(positions, [model["label"] for model in models], fontsize=13, color="#243244")
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 20, 40, 60, 80, 100], labels=[])
        ax.tick_params(axis="both", length=0, pad=10)
        ax.grid(axis="x", color="#d3dbe4", linewidth=.8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for position, model, value in zip(positions, models, values):
            label = (format_number(model["score"], 4) if task == "describe"
                     else f"{format_number(value, 2)}%")
            ax.text(value + 1.5, position, label, va="center", fontsize=13, color="#243244")
        fig.suptitle(title, y=.96, fontsize=21, fontweight="bold", color="#243244")
        fig.text(.5, .85, metric_label, ha="center", fontsize=13, color="#596778")
        fig.text(.5, .055, "Higher is better · Failures included",
                 ha="center", fontsize=11, color="#596778")
        fig.savefig(output / f"{figure}.png", dpi=180)
        fig.savefig(output / f"{figure}.svg", metadata={"Date": None})
        plt.close(fig)
        task_comparisons.append({
            "task": task, "title": title, "metric": metric, "metric_label": metric_label,
            "figure": figure, "bar_range": [0, 100], "score_multiplier": 100,
            "value_format": "native" if task == "describe" else "percent", "models": models,
        })

    figure_names = [profile["figure"] for profile in profiles]
    figure_names += [task["figure"] for task in task_comparisons]
    write_json(output / "manifest.json", {
        "generated_at": utc_now(), "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__, "radial_range": [0, 100], "score_multiplier": 100,
        "axes": [{"task": task, "metric": metric, "label": label, "metric_label": metric_label}
                 for task, metric, label, metric_label in AXES],
        "models": profiles,
        "task_comparisons": task_comparisons,
        "source_sha256": {f"benchmarks/src/{name}": digest(Path(__file__).with_name(name).read_bytes())
                          for name in ("generate_charts.py", "generate_report.py", "task_registry.py", "artifacts.py")},
        "reports_sha256": {str(path.relative_to(directory)): digest(path.read_bytes()) for path, _ in reports},
        "figures_sha256": {path.relative_to(output).as_posix(): digest(path.read_bytes())
                           for name in figure_names for suffix in ("png", "svg")
                           for path in [output / f"{name}.{suffix}"]},
    })
    generate_report(directory)
    return profiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("benchmarks/runs"))
    generate_charts(parser.parse_args().runs)
