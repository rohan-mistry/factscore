import logging
import os
from itertools import permutations
from typing import Dict, List, Set, Tuple, Union

import pysbd
from rich import print
from sumeval.metrics.rouge import RougeCalculator

from factsumm.utils.module_entity import load_ner, load_rel
from factsumm.utils.utils import Config
import spacy

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("flair").setLevel(logging.ERROR)

class FactSumm:

    def __init__(
        self,
        ner_model: str = None,
        rel_model: str = None,
        qg_model: str = None,
        qa_model: str = None,
        bert_score_model: str = None,
    ):
        """
        FactSumm object used to calculate Factual Consistency score of Abstractive Summarization model

        Args:
            ner_model (str, optional): NER model to be used (Flair or HuggingFace). Defaults to None.
            rel_model (str, optional): RE model to be used (HuggingFace). Defaults to None.
            qg_model (str, optional): QA model to be used (HuggingFace). Defaults to None.
            qa_model (str, optional): QG model to be used (HuggingFace). Defaults to None.
            bert_score_model (str, optional): BERTScore model to be used (HuggingFace). Defaults to None.

        """
        self.config = Config()
        self.segmenter = pysbd.Segmenter(language="en", clean=False)

        # NER, RE, QG, QA models supported by HuggingFace can be used (default can be found in `config.py`)
        self.ner = ner_model if ner_model is not None else self.config.NER_MODEL
        self.rel = rel_model if rel_model is not None else self.config.REL_MODEL

    def build_perm(
        self,
        lines: List[str],
        total_entities: Union[List[Dict], List[List[Dict]]],
    ) -> List:
        """
        Build entity permutations for Relation Extraction

        Args:
            lines (List[str]): segmented document lines
            total_entities (Union[List[Dict], List[List[Dict]]]): list of total entities

        Returns:
            List: list of permutations

        """
        total_perms = list()

        for line, line_entities in zip(lines, total_entities):
            line_perms = list(permutations(line_entities, 2))

            line_perms = [{
                "text":
                    line,
                "spans": [
                    (comb[0]["start"], comb[0]["end"]),
                    (comb[-1]["start"], comb[-1]["end"]),
                ],
                "ner": (comb[0]['entity'], comb[-1]['entity'])
            } for comb in line_perms]

            total_perms.append(line_perms)

        return total_perms

    def get_facts(self, lines: List[str], entities: List[List[Dict]]) -> Set:
        """
        Get fact triples using Relation Extraction model

        Args:
            lines (List[str]): segmented document lines
            entities (List[List[Dict]]): list of total entities

        Returns:
            Set: set of relation inferenced from permutations

        """
        perms = self.build_perm(lines, entities)
        triples = list()

        for perm in perms:
            triples.extend(self.rel(perm))

        return set(triples)

    def _segment(self, text: str) -> List[str]:
        """
        Segment input text into (possibly) multiple sentences

        Args:
            text (str): text to be segmented

        Returns:
            List[str]: list of segmented lines

        """
        return [line.strip() for line in self.segmenter.segment(text)]

    def _print_entities(self, mode: str, total_entities: List[List[Dict]]):
        # yapf:disable
        print(f"{mode.upper()} Entities")
        for i, line_entities in enumerate(total_entities):
            print(f'{i+1}: {[(entity["word"], entity["entity"]) for entity in line_entities]}')
        print()
        # yapf:enable

    def _print_facts(self, mode: str, facts: Set[Tuple]):
        print(f"{mode.upper()} Facts")
        for fact in facts:
            print(fact)
        print()

    def formatData(self, triple):
        triple[0] = self.format_entity(triple[0])
        triple[2] = self.format_entity(triple[2])

        return tuple(triple)

    def _filter_out(self, sources: Set, summaries: Set) -> Tuple[Set, Set]:
        """
        Filter out triples that don't share a subject and relation for comparability

        Args:
            sources (Set): set of triples from source
            summaries (Set): set of triples from summary

        Returns:
            Tuple[Set, Set]: filtered sources and summaries

        """

        #Rohan: Used for filtering common facts in source which are also present in summary, changed for cross pairing numeric data

        filtered_sources = set()
        for source in sources:
            sanitised_source = self.formatData(list(source))
            for summary in summaries:
                sanitised_summary = self.formatData(list(summary))
                if (source[1] == summary[1] and 
                    ((sanitised_source[0] == sanitised_summary[2] or sanitised_source[2] == sanitised_summary[0]) or (sanitised_source[0] == sanitised_summary[0] or sanitised_source[1] == sanitised_summary[1]))):
                    filtered_sources.add(source)
                    continue

        filtered_summary = set()

        #Rohan: Used for filtering common facts in summary which are also present in source, changed for cross pairing numeric data

        for summary in summaries:
            sanitised_summary = self.formatData(list(summary))
            for source in sources:
                sanitised_source = self.formatData(list(source))
                if (source[1] == summary[1] and 
                    ((sanitised_source[0] == sanitised_summary[2] or sanitised_source[2] == sanitised_summary[0]) or (sanitised_source[0] == sanitised_summary[0] or sanitised_source[1] == sanitised_summary[1]))):
                    filtered_summary.add(summary)
                    continue


        return filtered_sources, filtered_summary


    # Rohan: Used for extracting common and uncommon facts considering the special case of cross pairing numeric data
    def extract_common_uncommon(self, sources, summaries):
        common_facts = set()
        uncommon_facts = set()
        for summary in summaries:
            sanitised_summary = self.formatData(list(summary))
            found = False
            for source in sources:
                sanitised_source = self.formatData(list(source))
                if ((sanitised_source[0]==sanitised_summary[0] and sanitised_source[1]==sanitised_summary[1] and sanitised_source[2]==sanitised_summary[2]) or (sanitised_source[1] == sanitised_summary[1] and (sanitised_source[0] == sanitised_summary[2] and sanitised_source[2] == sanitised_summary[0]))):
                    common_facts.add(summary)
                    found = True
                    continue

            if not found:
                uncommon_facts.add(summary)

        return common_facts, uncommon_facts
    

    #Rohan: Check if summary nuneric data is substring of source numeric data
    def checkIfStringIsSubstring(self, word, list):
        for key in list:
            if word in key:
                return True
            
        return False
    
    def format_entity(self, word):
        return ''.join(chunk.lower() for chunk in word.split())
    
    def is_word_present(self, paragraph, target_word):
        # Check if the target word is present in the paragraph
        return target_word in paragraph
    
    def extract_spacy_entities(self, lines):
        result = list()
        nlp = spacy.load("en_core_web_sm")
        for line in lines:
            doc = nlp(line)
            line_entities = list()
            cache = {}
            for ent in doc.ents:
                if ent.text not in cache:
                    line_entities.append({
                        "word": ent.text
                    })
                    cache[ent.text] = 1

            result.append(line_entities)

        return result

    def prepareEntityList(self, lines, entities):
        ents_including_spacy = entities + self.extract_spacy_entities(lines)
        combined_entities = {self.format_entity(entity['word']): 1 for entity_line in ents_including_spacy for entity in entity_line}
        del ents_including_spacy

        return combined_entities


    def extract_facts(
        self,
        source: str,
        summary: str,
        verbose: bool = False,
        device: str = "cpu",
    ):
        """
        Extract (head_entity, relation, tail_entity) relation triple using NER & RE module

            See also https://arxiv.org/abs/1905.13322.pdf

        Args:
            source (str): original source
            summary (str): generated summary
            verbose (bool, optional): print verbose option. Defaults to False.
            device (str): device info

        """
        if isinstance(self.ner, str) and isinstance(self.rel, str):
            self.ner = load_ner(self.ner, device)
            self.rel = load_rel(self.rel, device)

        source_lines = self._segment(source)
        summary_lines = self._segment(summary)

        # extract per-line entities
        source_ents = self.ner(source_lines)
        summary_ents = self.ner(summary_lines)

        # extract entity-based triple: (head, relation, tail)
        source_facts = self.get_facts(source_lines, source_ents)
        summary_facts = self.get_facts(summary_lines, summary_ents)

        self._print_facts("Unfiltered source", source_facts)
        self._print_facts("Unfiltered summary", summary_facts)

        # filter out some facts
        source_facts, summary_facts = self._filter_out(
            source_facts,
            summary_facts,
        )

        common_facts, diff_facts = self.extract_common_uncommon(source_facts, summary_facts)

        #Rohan: penalty for uncommon numeric facts

        total_matched_entities = 0

        numeric_source_entities = self.prepareEntityList(source_lines, source_ents)
        numeric_summary_entities = self.prepareEntityList(summary_lines, summary_ents)

        print('entity source : ')
        print(numeric_source_entities)
        print('entity summary : ')
        print(numeric_summary_entities)

        total_summary_entities = len(numeric_summary_entities)

        unmatched_entities = {}

        source = source.lower()

        #Rohan: Calculate count of numeric summary data which matches with source numeric data
        for entity in numeric_summary_entities:
            if numeric_source_entities.get(entity) != None or self.checkIfStringIsSubstring(entity, numeric_source_entities) or self.is_word_present(source, entity):
                total_matched_entities += 1
            else:
                unmatched_entities[entity] = 1

           
        print('Not matching entities : ')
        print(unmatched_entities)

        if verbose:
            self._print_entities("source", source_ents)
            self._print_entities("summary", summary_ents)

            self._print_facts("Filtered source", source_facts)
            self._print_facts("Filtered summary", summary_facts)

            self._print_facts("common", common_facts)
            self._print_facts("diff", diff_facts)

        if total_summary_entities != 0:
            #Rohan: Calculate numeric fact score
            entity_score = total_matched_entities / total_summary_entities
        else:
            entity_score = 0.0

        if not summary_facts:
            fact_score = 0.0
        else:
            fact_score = len(common_facts) / len(summary_facts)

        print(f"Fact Match Score: {fact_score}")
        #Rohan: Take weighted average of original fact score and entity fact score
        
        FACT_WEIGHATGE = 0.3
        ENTITY_WEIGHTAGE = 0.7
        fact_score = FACT_WEIGHATGE * fact_score + ENTITY_WEIGHTAGE * entity_score

        print(f"Entity Score: {total_matched_entities} / {total_summary_entities} : {entity_score}")
        print(f"Final Fact Score: {fact_score}")

        return source_ents, summary_ents, fact_score
