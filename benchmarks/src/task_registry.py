"""Ordered task definitions shared by preparation, execution and reporting."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    operator: str
    title: str
    metrics: tuple[tuple[str, str], ...]
    primary_metric: str
    chart_metric_label: str
    geometry: str | None = None

    @property
    def variation(self):
        return self.geometry.title() if self.geometry else None

    @property
    def chart_title(self):
        return f"{self.title} ({self.variation})" if self.variation else self.title


# Primary chart metrics have a natural [0, 1] range, independent of observed models.
TASKS = {
    "classify": TaskSpec(
        dataset="cifar10", operator="Classify", title="Classification",
        metrics=(("accuracy_top1", "Top-1 accuracy"),),
        primary_metric="accuracy_top1", chart_metric_label="Accuracy",
    ),
    "detect": TaskSpec(
        dataset="coco128", operator="Detect", title="Detection",
        metrics=(("precision", "P@.5"), ("recall", "R@.5"), ("f1", "F1@.5")),
        primary_metric="f1", chart_metric_label="F1@0.5",
    ),
    "ground_box": TaskSpec(
        dataset="refcocog", operator="Ground", title="Grounding", geometry="box",
        metrics=(("accuracy_iou_0.5", "Acc@.5"), ("accuracy_iou_0.75", "Acc@.75"),
                 ("accuracy_iou_0.9", "Acc@.9"), ("mean_iou", "Mean IoU")),
        primary_metric="accuracy_iou_0.5", chart_metric_label="Acc@0.5",
    ),
    "ground_point": TaskSpec(
        dataset="refcocog", operator="Ground", title="Grounding", geometry="point",
        metrics=(("point_in_box_accuracy", "Point in box"),),
        primary_metric="point_in_box_accuracy", chart_metric_label="Accuracy",
    ),
    "attributes": TaskSpec(
        dataset="ovad", operator="Attributes", title="Attributes",
        metrics=(("accuracy", "Accuracy"), ("mean_attribute_accuracy", "Mean accuracy"),
                 ("macro_f1", "Macro F1")),
        primary_metric="accuracy", chart_metric_label="Accuracy",
    ),
    "answer": TaskSpec(
        dataset="vqav2", operator="Answer", title="VQA",
        metrics=(("vqa_accuracy", "VQA accuracy"),),
        primary_metric="vqa_accuracy", chart_metric_label="Accuracy",
    ),
    "describe": TaskSpec(
        dataset="coco_captions2017", operator="Describe", title="Captioning",
        metrics=(("cider", "CIDEr"), ("bleu_1", "BLEU-1"), ("bleu_2", "BLEU-2"),
                 ("bleu_3", "BLEU-3"), ("bleu_4", "BLEU-4"), ("rouge_l", "ROUGE-L")),
        primary_metric="rouge_l", chart_metric_label="ROUGE-L",
    ),
}

DATASETS = {name: spec.dataset for name, spec in TASKS.items()}
METRICS = {name: spec.metrics for name, spec in TASKS.items()}
AXES = [(name, spec.primary_metric, spec.chart_title, spec.chart_metric_label)
        for name, spec in TASKS.items()]
TASK_SECTIONS = [(title, [(name, spec.variation) for name, spec in TASKS.items() if spec.title == title])
                 for title in dict.fromkeys(spec.title for spec in TASKS.values())]
