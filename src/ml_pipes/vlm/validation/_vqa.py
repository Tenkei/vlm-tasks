"""VQA v2 answer normalization and ten-annotator consensus.

Normalization follows lmms-eval's VQA v2 wrapper, which calls punctuation and
number/article processing directly, without its extra word_tokenize step.
The fixed tables and punctuation rules intentionally preserve upstream behavior.
"""
# Adopted from lmms-eval: https://github.com/EvolvingLMMs-Lab/lmms-eval
# Copyright (c) 2025 LMMs-Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Original processor provenance:
# https://github.com/facebookresearch/mmf/blob/c46b3b3391275b4181567db80943473a89ab98ab/pythia/tasks/processors.py#L897
from __future__ import annotations

from collections.abc import Sequence
import re

ANSWER_TYPES = ("yes/no", "number", "other")
SCORER_METADATA = {
    "protocol": "VQA v2 ten-answer leave-one-out consensus",
    "normalization": "lmms-eval VQA v2 (always normalized)",
    "scorer_version": 1,
    "upstream_revision": "c7eaef373f4b1bd8a6b5f3e85c3391962c30068c",
    "upstream_url": "https://github.com/EvolvingLMMs-Lab/lmms-eval",
}


class _AnswerProcessor:
    CONTRACTIONS = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }

    NUMBER_MAP = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    ARTICLES = ["a", "an", "the"]
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]


def normalize_answer(answer: str) -> str:
    text = answer.replace("\n", " ").replace("\t", " ").strip()
    result = text
    for punctuation in _AnswerProcessor.PUNCTUATIONS:
        if (
            punctuation + " " in text or " " + punctuation in text
            or _AnswerProcessor.COMMA_STRIP.search(text) is not None
        ):
            result = result.replace(punctuation, "")
        else:
            result = result.replace(punctuation, " ")
    result = _AnswerProcessor.PERIOD_STRIP.sub("", result, re.UNICODE)
    words = (
        _AnswerProcessor.NUMBER_MAP.get(word, word)
        for word in result.lower().split()
    )
    return " ".join(
        _AnswerProcessor.CONTRACTIONS.get(word, word)
        for word in words if word not in _AnswerProcessor.ARTICLES
    )


def validate_targets(answers, answer_type, question_type) -> None:
    if (
        isinstance(answers, (str, bytes)) or not isinstance(answers, Sequence)
        or len(answers) != 10 or any(not isinstance(answer, str) for answer in answers)
    ):
        raise ValueError("VQA v2 requires exactly ten string answer votes")
    if not isinstance(answer_type, str) or answer_type not in ANSWER_TYPES:
        raise ValueError(f"Unsupported VQA v2 answer_type: {answer_type!r}")
    if not isinstance(question_type, str):
        raise ValueError("VQA v2 question_type must be a string")


def score_answer(answer: str, answers: Sequence[str]) -> float:
    references = tuple(normalize_answer(value) for value in answers)
    prediction = normalize_answer(answer)
    matches = references.count(prediction)
    # Hold out a single annotator by index, retaining all duplicate votes.
    return sum(min(1.0, (matches - (reference == prediction)) / 3) for reference in references) / 10
