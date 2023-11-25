import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

from transformers import pipeline


@dataclass
class Config:
    NER_MODEL: str = "flair/ner-english-ontonotes-fast"
    REL_MODEL: str = "studio-ousia/luke-large-finetuned-tacred"


def grouped_entities(entities: List[Dict]) -> List:
    """
    Group entities to concatenate BIO

    Args:
        entities (List[Dict]): list of inference entities

    Returns:
        List[Tuple]: list of grouped BIO scheme entities

    """

    def _remove_prefix(entity: str) -> str:
        if "-" in entity:
            entity = entity[2:]
        return entity

    def _append(lst: List, word: str, type: str, start: int, end: int):
        if prev_word != "":
            lst.append((word, type, start, end))

    result = list()

    prev_word = entities[0]["word"]
    prev_entity = entities[0]["entity"]
    prev_type = _remove_prefix(prev_entity)
    prev_start = entities[0]["start"]
    prev_end = entities[0]["end"]

    for pair in entities[1:]:
        word = pair["word"]
        entity = pair["entity"]
        type = _remove_prefix(entity)
        start = pair["start"]
        end = pair["end"]

        if "##" in word:
            prev_word += word
            prev_end = end
            continue

        if entity == prev_entity:
            if entity == "O":
                _append(result, prev_word, prev_type, prev_start, prev_end)
                result.append((word, type))
                prev_word = ""
                prev_start = start
                prev_end = end
            if "I-" in entity:
                prev_word += f" {word}"
                prev_end = end
        elif (entity != prev_entity) and ("I-" in entity) and (type != "O"):
            prev_word += f" {word}"
            prev_end = end
        else:
            _append(result, prev_word, prev_type, prev_start, prev_end)
            prev_word = word
            prev_type = type
            prev_start = start
            prev_end = end

        prev_entity = entity

    _append(result, prev_word, prev_type, prev_start, prev_end)

    cache = dict()
    dedup = list()

    for pair in result:
        if pair[1] == "O":
            continue

        if pair[0] not in cache:
            dedup.append({
                "word": pair[0].replace("##", ""),
                "entity": pair[1],
                "start": pair[2],
                "end": pair[3]
            })
            cache[pair[0]] = None
    return dedup
