"""Create a local two-model report from saved benchmark results, without inference."""
from __future__ import annotations

import argparse
from os.path import relpath
from pathlib import Path
from urllib.parse import quote

from .artifacts import digest, utc_now, write_json
from .generate_charts import COLORS, MODEL_LABELS, build_profiles, configure_plotting, draw_profile
from .generate_report import TABLE_HEADER, load_task_reports, summarize_task_report
from .task_registry import AXES


def model_catalog(reports):
    runs = {path.parent.name: report["model"] for path, report in reports}
    return dict(sorted(runs.items(), key=lambda item: (item[1], item[0])))


def select_runs(catalog, names):
    if len(names) != 2:
        raise ValueError("Provide exactly two model names. Use --list-models to see available names.")
    selected = []
    for name in names:
        matches = [run for run, model in catalog.items() if name.strip().casefold() in {
            run.casefold(), model.casefold(), MODEL_LABELS.get(model, model).casefold(),
            model.removeprefix("Qwen/").removeprefix("qwen.").casefold(),
        }]
        if not matches:
            available = ", ".join(f"{MODEL_LABELS.get(model, model)} ({run})"
                                  for run, model in catalog.items())
            raise ValueError(f"Unknown model {name!r}. Available models: {available or 'none'}")
        if len(matches) > 1:
            raise ValueError(f"Model {name!r} matches multiple runs; use a run ID: {', '.join(matches)}")
        selected.append(matches[0])
    if selected[0] == selected[1]:
        raise ValueError("Choose two different model runs.")
    return selected


def generate_comparison(directory, names, output=None):
    """Write one pair's README, radar chart and provenance without changing the overview."""
    directory = Path(directory).resolve()
    reports = load_task_reports(directory)
    catalog = model_catalog(reports)
    selected = select_runs(catalog, names)
    reports = [(path, report) for path, report in reports if path.parent.name in selected]
    profiles = {profile["run"]: profile for profile in build_profiles(reports)}
    colors = {run: index for index, run in enumerate(catalog)}
    models = [{"run": run, "model": catalog[run],
               "label": MODEL_LABELS.get(catalog[run], catalog[run]),
               "backend": profiles[run]["backend"], "scores": profiles[run]["scores"],
               "samples": profiles[run]["samples"], "color": COLORS[colors[run] % len(COLORS)]}
              for run in selected]
    output = Path(output or directory.parent / "reports/model-comparisons" / "--vs--".join(selected)).resolve()

    def link(path):
        return quote(Path(relpath(path, output)).as_posix(), safe="/")

    indexed = {(path.parent.name, report["task"]): (path, report) for path, report in reports}
    table = []
    for task, *_ in AXES:
        for run in selected:
            path, report = indexed[run, task]
            _, row = summarize_task_report(path, report, link(path))
            table.append(row)

    title = " vs ".join(model["label"] for model in models)
    benchmark_root = Path(__file__).resolve().parents[1]
    lines = [f"# {title}", "", "Generated locally from saved benchmark results.", ""]
    for model in models:
        run = model["run"]
        lines += [f"- **{model['label']}**: `{model['model']}` · "
                  f"[Run configuration]({link(directory / run / 'config.json')}) (`{run}`)"]
    lines += [
        "", f"![{title}](comparison.png)", "",
        "[PNG](comparison.png) · [SVG](comparison.svg)", "",
        "Both profiles use the same axes and fixed scale; higher is better. "
        "Failures are included. Polygon area is not an overall score.", "",
        "## Results", "",
    ]
    lines += TABLE_HEADER + table + [
        "", f"See the [table field guide]({link(benchmark_root / 'README.md')}#table-field-guide) "
        f"and [methodology]({link(benchmark_root / 'methodology.md')}) for metric definitions, "
        "usage accounting and evaluation settings. Backend and hardware differences can affect comparisons.", "",
    ]

    import matplotlib
    import numpy as np

    plt = configure_plotting()
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    try:
        fig.subplots_adjust(left=.23, right=.77, bottom=.15, top=.77)
        for model in models:
            draw_profile(ax, model, colors[model["run"]], comparison=True)
        fig.suptitle(title, y=.96, fontsize=18, fontweight="bold", color="#243244")
        fig.legend(loc="lower center", bbox_to_anchor=(.5, .08), ncol=2, frameon=False)
        fig.text(.5, .055, "Higher is better · Failures included",
                 ha="center", fontsize=11, color="#596778")
        fig.savefig(output / "comparison.png", dpi=180)
        fig.savefig(output / "comparison.svg", metadata={"Date": None})
    finally:
        plt.close(fig)
    (output / "README.md").write_text("\n".join(lines))
    sources = {path for path, _ in reports}
    sources.update(directory / run / "config.json" for run in selected)
    sources.update(path.with_suffix(".jsonl") for path, _ in reports if path.with_suffix(".jsonl").is_file())
    write_json(output / "manifest.json", {
        "generated_at": utc_now(), "models": models,
        "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
        "radial_range": [0, 100], "score_multiplier": 100,
        "axes": [{"task": task, "metric": metric, "label": label, "metric_label": metric_label}
                 for task, metric, label, metric_label in AXES],
        "source_sha256": {f"benchmarks/src/{name}": digest(Path(__file__).with_name(name).read_bytes())
                          for name in ("generate_comparison.py", "generate_charts.py", "generate_report.py",
                                       "task_registry.py", "artifacts.py")},
        "inputs_sha256": {path.relative_to(directory).as_posix(): digest(path.read_bytes())
                          for path in sorted(sources)},
        "artifacts_sha256": {name: digest((output / name).read_bytes())
                             for name in ("README.md", "comparison.png", "comparison.svg")},
    })
    return output / "README.md"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", metavar="MODEL", help="Two names from the model charts")
    parser.add_argument("--runs", type=Path, default=Path("benchmarks/runs"))
    parser.add_argument("--output", type=Path, help="Directory for this comparison's report and images")
    parser.add_argument("--list-models", action="store_true", help="List model names and run IDs, then exit")
    args = parser.parse_args(argv)
    try:
        if args.list_models:
            for run, model in model_catalog(load_task_reports(args.runs)).items():
                print(f"{MODEL_LABELS.get(model, model)}  [{run}]  {model}")
        else:
            print(f"Created {generate_comparison(args.runs, args.models, args.output)}")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
