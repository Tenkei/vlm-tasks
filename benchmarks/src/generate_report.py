"""Generate the comparison table and CSV solely from completed task reports."""
from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path
import re

from .artifacts import digest, utc_now, write_json
from .task_registry import DATASETS, METRICS, TASK_SECTIONS

TASK_ANCHORS = {task: (variation or title).lower().replace(" ", "-")
                for title, variations in TASK_SECTIONS for task, variation in variations}
TABLE_HEADER = [
    "| Model | Task / dataset | N | Validation metrics | Failure rate (at cap) | Input tokens / sample | Output tokens / sample | Estimated cost / 1,000 samples | Mean latency (s) |",
    "|---|---|---:|---|---:|---:|---:|---:|---:|",
]


def format_number(value, decimal_places):
    text = f"{value:.{decimal_places}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def model_anchors(models):
    """Match GitHub's heading fragments, including repeated model names."""
    anchors, used = {}, set()
    for model in models:
        label = model.get("label") or model["model"]
        slug = re.sub(r"[^\w -]", "", label.lower()).replace(" ", "-")
        anchor, suffix = slug, 0
        while anchor in used:
            suffix += 1
            anchor = f"{slug}-{suffix}"
        used.add(anchor)
        anchors[model["run"]] = anchor
    return anchors


def load_task_reports(directory):
    directory = Path(directory)
    reports = []
    for run in sorted(directory.iterdir()):
        if not run.is_dir() or not (run / "config.json").is_file():
            continue
        config = json.loads((run / "config.json").read_text())
        for task in config["tasks"]:
            path = run / f"{task}.json"
            if path.is_file():
                reports.append((path, json.loads(path.read_text())))
    order = {task: i for i, task in enumerate(DATASETS)}
    reports.sort(key=lambda item: (order[item[1]["task"]], item[1]["model"]))
    return reports


def chart_grid(figures, links=None):
    lines = ["<table>"]
    for start in range(0, len(figures), 4):
        lines += ["  <tr>"]
        group = figures[start:start + 4]
        for name, title, description in group:
            image = escape(f"reports/charts/{name}.png", quote=True)
            vector = escape(f"reports/charts/{name}.svg", quote=True)
            alt = escape(description, quote=True)
            target = escape(links[name], quote=True) if links else image
            lines += [
                '    <td width="25%" align="center">',
                f'      <a href="{target}"><strong>{escape(title)}</strong></a><br>',
                f'      <a href="{target}"><img src="{image}" alt="{alt}" width="220"></a><br>',
                f'      <a href="{image}">PNG</a> · <a href="{vector}">SVG</a>',
                "    </td>",
            ]
        lines += ['    <td width="25%"></td>'] * (4 - len(group))
        lines += ["  </tr>"]
    return lines + ["</table>", ""]


def comparison_table(comparisons):
    models = comparisons[0]["models"]
    anchors = model_anchors(models)
    lines = [
        "Main validation metric per task; **best observed values** are bold. "
        "Failures are included. Caption ROUGE-L uses its native scale; other values are percentages. "
        "Select a model or task name for detailed metrics and usage.", "",
        "| Task | Main metric | " + " | ".join(
            f"[{model['label']}](#{anchors[model['run']]})" for model in models) + " |",
        "|---|---|" + "---:|" * len(models),
    ]
    for comparison in comparisons:
        scores = {model["run"]: model["score"] for model in comparison["models"]}
        best = max(scores.values())
        values = []
        for model in models:
            score = scores[model["run"]]
            value = (format_number(score, 4) if comparison["value_format"] == "native"
                     else f"{format_number(100 * score, 2)}%")
            values.append(f"**{value}**" if score == best else value)
        metric = dict(METRICS[comparison["task"]])[comparison["metric"]]
        link = f"[{comparison['title']}](#{TASK_ANCHORS[comparison['task']]})"
        lines.append(f"| {link} | {metric} | " + " | ".join(values) + " |")
    return lines + [""]


def model_sections(model_rows, charts):
    models = charts.get("models", [])
    if not models:
        return []
    labels = {model["run"]: model["label"]
              for task in charts.get("task_comparisons", []) for model in task["models"]}
    anchors = model_anchors([{**model, "label": labels.get(model["run"], model["model"])}
                             for model in models])
    lines = [
        "## Model profiles", "",
        "**Model view:** each radar chart summarizes one model across the benchmark tasks, "
        "showing its strengths and weaknesses. Select a chart or expand a model below "
        "for full metrics, usage, cost and latency.", "",
    ]
    lines += chart_grid([(model["figure"], labels.get(model["run"], model["model"]),
                          f"Model profile for {model['model']}") for model in models],
                        links={model["figure"]: f"#{anchors[model['run']]}" for model in models})
    lines += [
        "All profiles use the same axes and scale; higher is better. Failures are included. "
        "Polygon area is not an overall score. "
        "[Chart method](methodology.md#model-profile-charts)", "",
    ]
    for model in models:
        run = model["run"]
        label = escape(labels.get(run, model["model"]))
        image = escape(f"reports/charts/{model['figure']}.png", quote=True)
        alt = escape(f"Model profile for {model['model']}", quote=True)
        lines += [
            f'<a id="{escape(f"model-{run}", quote=True)}"></a>', "",
            f"### {label}", "",
            "<details>", "<summary>View chart, metrics and usage</summary>", "",
            f'<a href="{image}"><img src="{image}" alt="{alt}" width="720"></a>', "",
        ]
        lines += TABLE_HEADER + model_rows[run] + ["", "</details>", ""]
    return lines


def task_sections(table_rows, charts):
    comparisons = {task["task"]: task for task in charts.get("task_comparisons", [])}
    lines = ["## Task comparisons", ""]
    if comparisons:
        lines += [
            "**Task view:** each bar chart compares models on the same task using its main "
            "validation metric. Select a chart or expand a task below for all metrics, "
            "usage, cost and latency.", "",
        ]
        lines += chart_grid([
            (comparison["figure"], comparison["title"], f"Model comparison: {comparison['title']}")
            for comparison in comparisons.values()
        ], links={comparison["figure"]: f"#{TASK_ANCHORS[comparison['task']]}"
                  for comparison in comparisons.values()})
    else:
        lines += ["Validation metrics and usage by task. Higher validation scores are better; failures are included.", ""]
    for title, variations in TASK_SECTIONS:
        if not any(table_rows.get(task) for task, _ in variations):
            continue
        has_variations = any(variation for _, variation in variations)
        summary = "View variations" if has_variations else "View chart, metrics and usage"
        lines += [f"### {title}", "", "<details>", f"<summary>{summary}</summary>", ""]
        for task, variation in variations:
            if not table_rows.get(task):
                continue
            if variation:
                lines += [f"#### {variation}", "", "<details>",
                          "<summary>View chart, metrics and usage</summary>", ""]
            if task in comparisons:
                comparison = comparisons[task]
                image = escape(f"reports/charts/{comparison['figure']}.png", quote=True)
                alt = escape(f"Model comparison: {comparison['title']}", quote=True)
                lines += [
                    f'<a href="{image}"><img src="{image}" alt="{alt}" width="720"></a>', "",
                ]
            lines += TABLE_HEADER + table_rows[task] + [""]
            if variation:
                lines += ["</details>", ""]
        lines += ["</details>", ""]
    return lines


def generate_report(directory):
    """Build the benchmark overview and report artifacts from saved task reports."""
    directory = Path(directory)
    reports = load_task_reports(directory)
    output = directory.parent / "reports"
    output.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VLM benchmarks: by model and task", "",
        "Generated by `python -m benchmarks.src.generate_report`. [Methodology](methodology.md) · "
        "[Code](src/) · [CSV](reports/metrics.csv) · [Table fields](#table-field-guide)", "",
    ]
    notes = [
        "- **Model**: the evaluated model identifier; model columns in the overview identify "
        "the same runs. Their headings link to model details. Hardware, provider settings "
        "and regions are recorded in the run metadata and can affect comparisons.", "",
        "- **Task / dataset**, **Task**: the task, its variation and evaluation data. Detail-table "
        "links open saved reports; overview task names open the matching task details. "
        "Classification uses CIFAR-10 test (`val` alias), detection uses COCO128 training images, "
        "and OVAD uses its benchmark (`val` alias). Grounding, VQA and captions use validation "
        "splits. `ClassifyMultiLabel` has no built-in validation dataset and is excluded.", "",
        "- **N**: the number of evaluated samples, including failures. Token means, cost projections, "
        "failure rates and cap rates use this count. Samples can share images.", "",
        "- **Validation metrics**, **Main metric**: task quality scores from the validators. "
        "Main metric selects the score used in the overview and charts. Higher is better; "
        "ratios are percentages and caption scores use their native scale. "
        "`Acc` means accuracy: `Top-1 Acc` measures classification, `Mean Acc` averages "
        "attribute-group accuracies, `VQA Acc` uses answer consensus, and `Point-in-box Acc` "
        "counts predicted points inside target boxes. For Grounding (Box), `Acc@0.5`, "
        "`Acc@0.75` and `Acc@0.9` require IoU ≥ the indicated threshold. "
        "Detection's `P`, `R` and `F1` use the indicated IoU matching threshold. Bold overview values "
        "mark the best observed result within a task. These small subsets provide a preliminary "
        "comparison; datasets and scoring differ by task. See the [scoring rules](../docs/validation.md) "
        "and [benchmark method](methodology.md) for metric definitions and subset limitations.", "",
        "- **Failure rate (at cap)**: the percentage of samples that failed, followed in parentheses "
        "by the percentage of responses reaching the configured output limit (4,096 tokens in "
        "these runs). Both rates use N. A cap hit can be a valid response; inspect raw output "
        "to distinguish truncation from completion exactly at the limit.", "",
        "- **Input tokens / sample**, **Output tokens / sample**: recorded token totals divided "
        "by N, rounded to whole tokens. Input includes instructions, schemas and image tokens; "
        "output includes reasoning and invalid generated responses. Counts depend on each "
        "model's tokenizer and provider image preprocessing.", "",
        "- **Estimated cost / 1,000 samples**: estimated USD cost divided by N and multiplied "
        "by 1,000, including failed samples. This projects observed usage without assuming "
        "batch discounts. Hosted costs are estimates, not invoices; local monetary cost is "
        "unmeasured. `—` means cost is unavailable and `(partial)` means cost coverage is incomplete.", "",
        "- **Mean latency (s)**: mean seconds per evaluated sample, including image encoding, "
        "preprocessing, generation and parsing. Includes the first inference call and hosted "
        "network/service time; excludes model loading and dataset preparation.", "",
    ]
    rows = []
    table_rows = {}
    model_rows = {}
    for path, report in reports:
        task, m, usage = report["task"], report["metrics"], report["usage"]
        config = json.loads((path.parent / "config.json").read_text())
        calls_path = path.with_suffix(".jsonl")
        at_cap = None
        if calls_path.is_file():
            calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
            at_cap = sum(g["output_tokens"] >= config["max_new_tokens"]
                         for call in calls for g in call["generations"])
        scores = []
        for key, label in METRICS[task]:
            if key in m:
                value = format_number(m[key], 4) if task == "describe" else f"{format_number(100 * m[key], 2)}%"
                scores.append(f"{label} {value}")
        latency = report["speed_ms_per_sample"]["inference"] / 1000
        relative = path.relative_to(directory).as_posix()
        samples = m["samples"]
        failure_rate = m["failures"] / samples if samples else None
        cap_rate = at_cap / samples if at_cap is not None and samples else None
        input_per_sample = usage["input_tokens"] / samples if samples else None
        output_per_sample = usage["output_tokens"] / samples if samples else None
        cost = usage.get("estimated_cost_usd")
        cost_per_sample = float(cost) / samples if cost is not None and samples else None
        cost_per_1k_samples = cost_per_sample * 1000 if cost_per_sample is not None else None
        cost_text = format_number(cost_per_1k_samples, 3) if cost_per_1k_samples is not None else "—"
        if cost is not None and usage.get("estimated_cost_complete") is False:
            cost_text += " (partial)"
        failure_text = f"{format_number(100 * failure_rate, 2)}%" if failure_rate is not None else "—"
        cap_text = f"{format_number(100 * cap_rate, 2)}%" if cap_rate is not None else "unknown"
        input_text = f"{input_per_sample:,.0f}" if input_per_sample is not None else "—"
        output_text = f"{output_per_sample:,.0f}" if output_per_sample is not None else "—"
        table_row = (
            f"| {report['model']} | [{task} / {report['dataset']}](runs/{relative}) | "
            f"{samples} | {'; '.join(scores)} | {failure_text} ({cap_text}) | "
            f"{input_text} | {output_text} | {cost_text} | {format_number(latency, 3)} |")
        table_rows.setdefault(task, []).append(table_row)
        model_rows.setdefault(path.parent.name, []).append(table_row)
        row = {"run": path.parent.name, "model": report["model"], "revision": report["revision"],
               "task": task, "dataset": report["dataset"], "device": report["device"], "dtype": report["dtype"],
               "backend": report.get("backend", "transformers"), "region": report.get("region"),
               **m, **{f"usage_{k}": v for k, v in usage.items()},
               "failure_rate": failure_rate,
               "max_new_tokens": config["max_new_tokens"], "responses_at_token_cap": at_cap,
               "responses_at_token_cap_per_sample": cap_rate,
               "usage_input_tokens_per_sample": input_per_sample,
               "usage_output_tokens_per_sample": output_per_sample,
               "usage_estimated_cost_usd_per_sample": cost_per_sample,
               "usage_estimated_cost_usd_per_1k_samples": cost_per_1k_samples,
               "inference_seconds_per_sample": latency, "elapsed_seconds": report["elapsed_seconds"],
               "dataset_fingerprint_sha256": report["dataset_fingerprint_sha256"]}
        rows.append(row)
    report_hashes = {str(path.relative_to(directory)): digest(path.read_bytes()) for path, _ in reports}
    chart_manifest = output / "charts" / "manifest.json"
    current_charts = {}
    if chart_manifest.is_file():
        charts = json.loads(chart_manifest.read_text())
        if charts["reports_sha256"] == report_hashes:
            current_charts = charts
    if current_charts.get("task_comparisons"):
        lines += comparison_table(current_charts["task_comparisons"])
    lines += model_sections(model_rows, current_charts)
    lines += task_sections(table_rows, current_charts)
    lines += ["## Table field guide", ""] + notes
    directory.mkdir(parents=True, exist_ok=True)
    (directory.parent / "README.md").write_text("\n".join(lines))
    write_json(output / "report-manifest.json", {
        "generated_at": utc_now(),
        "summarizer_sha256": digest(Path(__file__).read_bytes()),
        "source_sha256": {f"benchmarks/src/{name}": digest(Path(__file__).with_name(name).read_bytes())
                          for name in ("generate_report.py", "task_registry.py", "artifacts.py")},
        "reports_sha256": report_hashes,
    })
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if rows:
        with (output / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("benchmarks/runs"))
    generate_report(parser.parse_args().runs)
