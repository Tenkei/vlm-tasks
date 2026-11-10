"""Local COCO caption metrics on PTB-tokenized text.

Equations and numerical conventions follow coco-caption (BSD licensed; see
NOTICE.caption). No VQA answer normalization is applied here.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import math


SCORER_METADATA = {
    "scorer": "local COCO caption metrics",
    "reference_implementation": "salaniz/pycocoevalcap@a24f74c408c918f1f4ec34e9514bc8a76ce41ffd",
    "cider_variant": "COCO clipped TF-IDF with Gaussian length penalty",
    "cider_ngram_order": 4, "cider_sigma": 6.0, "cider_scale": 10.0,
    "idf_reference_scope": "selected_images",
    "bleu_reference_length": "closest, shorter on ties",
    "rouge_l_beta": 1.2, "score_scale": "native (not percentages)",
    "failed_candidate": "empty string; references retained",
}


def validate_captions(captions: Sequence[str]) -> tuple[str, ...]:
    if (
        isinstance(captions, (str, bytes)) or not isinstance(captions, Sequence)
        or not captions
        or any(not isinstance(text, str) or not text.strip() for text in captions)
    ):
        raise ValueError("captions must be a nonempty sequence of nonempty strings")
    return tuple(captions)


def _ngrams(words):
    return Counter(tuple(words[i:i + n]) for n in range(1, 5) for i in range(len(words) - n + 1))


def _lcs(a, b):
    # Keep only one row; captions can vary widely in length.
    if len(a) < len(b):
        a, b = b, a
    row = [0] * (len(b) + 1)
    for word in a:
        previous = 0
        for j, other in enumerate(b, 1):
            old = row[j]
            row[j] = previous + 1 if word == other else max(row[j], row[j - 1])
            previous = old
    return row[-1]


def score_captions(candidates: Sequence[str], references: Sequence[Sequence[str]]) -> dict[str, float]:
    """Score one candidate per image, with all references retained in the IDF."""
    if not candidates or len(candidates) != len(references) or any(not refs for refs in references):
        raise ValueError("Caption scoring requires aligned candidates and nonempty reference sets")
    candidates = [text.split() for text in candidates]
    references = [[text.split() for text in refs] for refs in references]
    candidate_counts = [_ngrams(words) for words in candidates]
    reference_counts = [[_ngrams(words) for words in refs] for refs in references]
    df = Counter(gram for refs in reference_counts for gram in set().union(*refs))
    log_images = math.log(len(references))

    def vector(counts):
        weights = {gram: count * (log_images - math.log(max(df[gram], 1))) for gram, count in counts.items()}
        norms = [math.sqrt(sum(weight * weight for gram, weight in weights.items() if len(gram) == n)) for n in range(1, 5)]
        length = sum(count for gram, count in counts.items() if len(gram) == 2)
        return weights, norms, length

    correct, guesses = [0] * 4, [0] * 4
    candidate_length = reference_length = 0
    cider_total = rouge_total = 0.0
    for words, refs, counts, ref_counts in zip(candidates, references, candidate_counts, reference_counts):
        candidate_length += len(words)
        reference_length += min((len(ref) for ref in refs), key=lambda length: (abs(length - len(words)), length))
        maximum = Counter()
        for ref in ref_counts:
            maximum |= ref
        for n in range(1, 5):
            guesses[n - 1] += max(len(words) - n + 1, 0)
            correct[n - 1] += sum(min(count, maximum[gram]) for gram, count in counts.items() if len(gram) == n)

        weights, norms, length = vector(counts)
        similarity = 0.0
        for ref in ref_counts:
            ref_weights, ref_norms, ref_length = vector(ref)
            penalty = math.exp(-((length - ref_length) ** 2) / (2 * 6.0 ** 2))
            for n in range(1, 5):
                dot = sum(min(weight, ref_weights.get(gram, 0.0)) * ref_weights.get(gram, 0.0)
                          for gram, weight in weights.items() if len(gram) == n)
                if norms[n - 1] and ref_norms[n - 1]:
                    dot /= norms[n - 1] * ref_norms[n - 1]
                similarity += dot * penalty
        cider_total += similarity / (4 * len(refs)) * 10.0

        precision = recall = 0.0
        for ref in refs:
            overlap = _lcs(words, ref)
            precision = max(precision, overlap / len(words) if words else 0.0)
            recall = max(recall, overlap / len(ref) if ref else 0.0)
        if precision and recall:
            rouge_total += (1 + 1.2 ** 2) * precision * recall / (recall + 1.2 ** 2 * precision)

    ratio = (candidate_length + 1e-15) / (reference_length + 1e-9)
    penalty = math.exp(1 - 1 / ratio) if ratio < 1 else 1.0
    scores = {"cider": cider_total / len(candidates)}
    product = 1.0
    for n in range(1, 5):
        product *= (correct[n - 1] + 1e-15) / (guesses[n - 1] + 1e-9)
        scores[f"bleu_{n}"] = product ** (1 / n) * penalty
    scores["rouge_l"] = rouge_total / len(candidates)
    return scores
