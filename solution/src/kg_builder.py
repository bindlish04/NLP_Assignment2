"""
Task 2: Knowledge graph construction.

Entities are extracted with (1) a domain lexicon mapped to ontology classes and
(2) spaCy NER. Relationships come from regex patterns and typed co-occurrence
within the same sentence or document.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import networkx as nx

from .config import MIN_KG_ENTITIES, MIN_KG_RELATIONSHIPS
from .ontology import DomainOntology

# --- Domain lexicon: surface phrase -> ontology entity class ---
ENTITY_LEXICON: dict[str, str] = {
    "global warming": "ClimatePhenomenon",
    "climate change": "ClimatePhenomenon",
    "sea level rise": "ClimatePhenomenon",
    "ocean acidification": "ClimatePhenomenon",
    "extreme weather": "ClimatePhenomenon",
    "deforestation": "ClimatePhenomenon",
    "desertification": "ClimatePhenomenon",
    "biodiversity loss": "ClimatePhenomenon",
    "coral bleaching": "ClimatePhenomenon",
    "glacial retreat": "ClimatePhenomenon",
    "arctic sea ice decline": "ClimatePhenomenon",
    "heat wave": "ClimatePhenomenon",
    "heat waves": "ClimatePhenomenon",
    "drought": "ClimatePhenomenon",
    "flooding": "ClimatePhenomenon",
    "wildfire": "ClimatePhenomenon",
    "wildfires": "ClimatePhenomenon",
    "climate feedback": "ClimatePhenomenon",
    "climate feedbacks": "ClimatePhenomenon",
    "urban heat island": "ClimatePhenomenon",
    "permafrost thaw": "ClimatePhenomenon",
    "climate variability": "ClimatePhenomenon",
    "climate emergency": "ClimatePhenomenon",
    "radiative forcing": "ClimatePhenomenon",
    "greenhouse effect": "ClimatePhenomenon",
    "attribution of recent climate change": "ClimatePhenomenon",
    "carbon dioxide": "GreenhouseGas",
    "co2": "GreenhouseGas",
    "methane": "GreenhouseGas",
    "ch4": "GreenhouseGas",
    "nitrous oxide": "GreenhouseGas",
    "n2o": "GreenhouseGas",
    "greenhouse gas": "GreenhouseGas",
    "greenhouse gases": "GreenhouseGas",
    "fluorinated gas": "GreenhouseGas",
    "fluorinated gases": "GreenhouseGas",
    "carbon emissions": "GreenhouseGas",
    "carbon emission": "GreenhouseGas",
    "carbon footprint": "GreenhouseGas",
    "atmospheric co2": "GreenhouseGas",
    "sf6": "GreenhouseGas",
    "cfc": "GreenhouseGas",
    "hfcs": "GreenhouseGas",
    "paris agreement": "PolicyInstrument",
    "kyoto protocol": "PolicyInstrument",
    "carbon tax": "PolicyInstrument",
    "emissions trading": "PolicyInstrument",
    "carbon capture": "PolicyInstrument",
    "carbon capture and storage": "PolicyInstrument",
    "cap and trade": "PolicyInstrument",
    "carbon pricing": "PolicyInstrument",
    "carbon offset": "PolicyInstrument",
    "carbon offsets": "PolicyInstrument",
    "net zero": "PolicyInstrument",
    "net-zero": "PolicyInstrument",
    "climate mitigation": "PolicyInstrument",
    "climate adaptation": "PolicyInstrument",
    "renewable portfolio standard": "PolicyInstrument",
    "feed-in tariff": "PolicyInstrument",
    "montreal protocol": "PolicyInstrument",
    "climate justice": "PolicyInstrument",
    "ipcc": "Organization",
    "intergovernmental panel on climate change": "Organization",
    "unfccc": "Organization",
    "world meteorological organization": "Organization",
    "united nations": "Organization",
    "european union": "Organization",
    "nasa": "Organization",
    "noaa": "Organization",
    "world bank": "Organization",
    "iea": "Organization",
    "international energy agency": "Organization",
    "green climate fund": "Organization",
    "arctic": "GeographicRegion",
    "antarctic": "GeographicRegion",
    "amazon rainforest": "GeographicRegion",
    "amazon": "GeographicRegion",
    "indian ocean": "GeographicRegion",
    "himalayas": "GeographicRegion",
    "greenland": "GeographicRegion",
    "antarctica": "GeographicRegion",
    "sahara": "GeographicRegion",
    "great barrier reef": "GeographicRegion",
    "pacific ocean": "GeographicRegion",
    "atlantic ocean": "GeographicRegion",
    "india": "GeographicRegion",
    "china": "GeographicRegion",
    "united states": "GeographicRegion",
    "europe": "GeographicRegion",
    "africa": "GeographicRegion",
    "asia": "GeographicRegion",
    "tropics": "GeographicRegion",
    "siberia": "GeographicRegion",
    "solar power": "EnergySource",
    "solar energy": "EnergySource",
    "wind power": "EnergySource",
    "wind energy": "EnergySource",
    "coal": "EnergySource",
    "natural gas": "EnergySource",
    "oil": "EnergySource",
    "fossil fuel": "EnergySource",
    "fossil fuels": "EnergySource",
    "renewable energy": "EnergySource",
    "renewables": "EnergySource",
    "electric vehicle": "EnergySource",
    "electric vehicles": "EnergySource",
    "hydropower": "EnergySource",
    "hydroelectric power": "EnergySource",
    "nuclear power": "EnergySource",
    "geothermal energy": "EnergySource",
    "biofuel": "EnergySource",
    "biofuels": "EnergySource",
    "biomass": "EnergySource",
    "petroleum": "EnergySource",
    "crude oil": "EnergySource",
    "lng": "EnergySource",
}

# Map spaCy entity labels to our ontology (only labels we trust for this domain).
SPACY_LABEL_MAP = {
    "ORG": "Organization",
    "GPE": "GeographicRegion",
    "LOC": "GeographicRegion",
    "NORP": "Organization",
    "PRODUCT": "EnergySource",
    "EVENT": "ClimatePhenomenon",
}

# Regex patterns for explicit relation phrases in text.
RELATION_PATTERN_SPECS: list[tuple[str, str]] = [
    (r"(\w[\w\s\-]{1,40}?)\s+causes?\s+(\w[\w\s\-]{1,40})", "causes"),
    (r"(\w[\w\s\-]{1,40}?)\s+contributes?\s+to\s+(\w[\w\s\-]{1,40})", "contributes_to"),
    (r"(\w[\w\s\-]{1,40}?)\s+mitigated\s+by\s+(\w[\w\s\-]{1,40})", "mitigated_by"),
    (r"(\w[\w\s\-]{1,40}?)\s+regulated\s+by\s+(\w[\w\s\-]{1,40})", "regulated_by"),
    (r"(\w[\w\s\-]{1,40}?)\s+located\s+in\s+(\w[\w\s\-]{1,40})", "located_in"),
    (r"(\w[\w\s\-]{1,40}?)\s+emits?\s+(\w[\w\s\-]{1,40})", "emits"),
    (r"(\w[\w\s\-]{1,40}?)\s+leads?\s+to\s+(\w[\w\s\-]{1,40})", "causes"),
    (r"(\w[\w\s\-]{1,40}?)\s+results?\s+in\s+(\w[\w\s\-]{1,40})", "causes"),
    (r"(\w[\w\s\-]{1,40}?)\s+reduces?\s+(\w[\w\s\-]{1,40})", "mitigated_by"),
    (r"(\w[\w\s\-]{1,40}?)\s+increases?\s+(\w[\w\s\-]{1,40})", "causes"),
]
COMPILED_RELATION_PATTERNS = [(re.compile(p), rel) for p, rel in RELATION_PATTERN_SPECS]

# When two entity types co-occur, infer these relation types (heuristic fallback).
CO_OCCURRENCE_RELATIONS = [
    ("GreenhouseGas", "ClimatePhenomenon", "causes"),
    ("EnergySource", "GreenhouseGas", "emits"),
    ("PolicyInstrument", "ClimatePhenomenon", "mitigated_by"),
    ("Organization", "ClimatePhenomenon", "contributes_to"),
    ("ClimatePhenomenon", "GeographicRegion", "located_in"),
    ("EnergySource", "ClimatePhenomenon", "causes"),
    ("Organization", "PolicyInstrument", "regulated_by"),
    ("PolicyInstrument", "GreenhouseGas", "mitigated_by"),
]

# Longest phrases first so "carbon dioxide" wins over "carbon".
SORTED_LEXICON = sorted(ENTITY_LEXICON.items(), key=lambda item: -len(item[0]))

_NLP = None


def _get_nlp():
    """Load spaCy once and reuse across documents (expensive to reload)."""
    global _NLP
    if _NLP is None:
        import spacy

        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download

            download("en_core_web_sm")
            _NLP = spacy.load("en_core_web_sm")
    return _NLP


@dataclass
class KnowledgeGraph:
    """Container for the NetworkX graph, entity registry, and triple list."""

    graph: nx.MultiDiGraph
    entities: dict[str, dict]
    triples: list[tuple[str, str, str]]

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def num_relationships(self) -> int:
        return len(self.triples)

    def stats(self) -> dict:
        type_counts: dict[str, int] = defaultdict(int)
        rel_counts: dict[str, int] = defaultdict(int)
        for meta in self.entities.values():
            type_counts[meta["type"]] += 1
        for _, rel, _ in self.triples:
            rel_counts[rel] += 1
        return {
            "entities": self.num_entities,
            "relationships": self.num_relationships,
            "entity_types": dict(type_counts),
            "relationship_types": dict(rel_counts),
        }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _register_entity(
    entities: dict[str, dict],
    graph: nx.MultiDiGraph,
    name: str,
    etype: str,
    source: str,
    source_map: dict[str, set[str]],
) -> str | None:
    """Add an entity node if the span is valid; returns normalized key or None."""
    key = _normalize(name)
    if len(key) < 2 or len(key.split()) > 8:
        return None
    entities.setdefault(key, {"name": key, "type": etype, "label": name.strip().title()})
    graph.add_node(key, type=etype, label=name.strip().title())
    source_map[key].add(source)
    return key


def _match_lexicon_entities(text: str) -> list[tuple[str, str]]:
    """Return non-overlapping (phrase, type) pairs found in text."""
    lower = text.lower()
    matches: list[tuple[str, str, int]] = []
    for phrase, etype in SORTED_LEXICON:
        for match in re.finditer(re.escape(phrase), lower):
            matches.append((phrase, etype, match.start()))
    matches.sort(key=lambda x: x[2])

    deduped: list[tuple[str, str]] = []
    seen_spans: list[tuple[int, int]] = []
    for name, etype, start in matches:
        end = start + len(name)
        if any(not (end <= s or start >= e) for s, e in seen_spans):
            continue
        seen_spans.append((start, end))
        deduped.append((name, etype))
    return deduped


def _extract_spacy_entities_batch(texts: list[str]) -> list[list[tuple[str, str]]]:
    """Run NER on multiple documents efficiently via nlp.pipe."""
    nlp = _get_nlp()
    results: list[list[tuple[str, str]]] = []
    for doc in nlp.pipe(texts, batch_size=8):
        found = []
        for ent in doc.ents:
            etype = SPACY_LABEL_MAP.get(ent.label_)
            if etype:
                found.append((ent.text, etype))
        results.append(found)
    return results


def _extract_pattern_relations(sentence: str) -> list[tuple[str, str, str]]:
    sent = _normalize(sentence)
    triples: list[tuple[str, str, str]] = []
    for pattern, rel in COMPILED_RELATION_PATTERNS:
        for match in pattern.finditer(sent):
            subj, obj = _normalize(match.group(1)), _normalize(match.group(2))
            if len(subj.split()) <= 6 and len(obj.split()) <= 6:
                triples.append((subj, rel, obj))
    return triples


def _infer_cooccurrence_relations(pairs: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    for name_a, type_a in pairs:
        for name_b, type_b in pairs:
            if name_a == name_b:
                continue
            for dom, rng, rel in CO_OCCURRENCE_RELATIONS:
                if type_a == dom and type_b == rng:
                    triples.append((name_a, rel, name_b))
    return triples


def _add_triple(
    triples: set[tuple[str, str, str]],
    graph: nx.MultiDiGraph,
    subj: str,
    rel: str,
    obj: str,
    source: str = "",
) -> None:
    if subj and obj and subj != obj:
        triples.add((subj, rel, obj))
        if source:
            graph.add_edge(subj, obj, relation=rel, source=source)


def build_knowledge_graph(
    documents: Iterable[dict],
    ontology: DomainOntology,
    min_entities: int = MIN_KG_ENTITIES,
    min_relationships: int = MIN_KG_RELATIONSHIPS,
) -> KnowledgeGraph:
    """
    Build a directed multigraph from all domain documents.

    Pipeline:
      1. Lexicon + spaCy entity spotting
      2. Pattern-based relation extraction per sentence
      3. Typed co-occurrence within sentences and documents
    """
    doc_list = list(documents)
    graph = nx.MultiDiGraph()
    entities: dict[str, dict] = {}
    triples: set[tuple[str, str, str]] = set()
    source_map: dict[str, set[str]] = defaultdict(set)
    doc_entity_keys: dict[str, set[str]] = defaultdict(set)

    # Batch NER across documents (much faster than one call per sentence).
    spacy_entities = _extract_spacy_entities_batch([d["text"][:100_000] for d in doc_list])

    for doc, ner_found in zip(doc_list, spacy_entities):
        text = doc["text"]
        doc_id = doc.get("file_name", doc.get("title", "unknown"))

        def track(name: str, etype: str) -> None:
            key = _register_entity(entities, graph, name, etype, doc_id, source_map)
            if key:
                doc_entity_keys[doc_id].add(key)

        for name, etype in _match_lexicon_entities(doc.get("title", "")):
            track(name, etype)
        for name, etype in ner_found:
            track(name, etype)

        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if len(sentence.split()) < 6:
                continue

            found = _match_lexicon_entities(sentence)
            sentence_keys: list[tuple[str, str]] = []
            for name, etype in found:
                key = _register_entity(entities, graph, name, etype, doc_id, source_map)
                if key:
                    doc_entity_keys[doc_id].add(key)
                    sentence_keys.append((key, etype))

            for subj, rel, obj in _extract_pattern_relations(sentence):
                if subj not in entities and subj in ENTITY_LEXICON:
                    track(subj, ENTITY_LEXICON[subj])
                if obj not in entities and obj in ENTITY_LEXICON:
                    track(obj, ENTITY_LEXICON[obj])
                if subj in entities and obj in entities:
                    _add_triple(triples, graph, subj, rel, obj, doc_id)

            for subj, rel, obj in _infer_cooccurrence_relations(sentence_keys):
                _add_triple(triples, graph, subj, rel, obj, doc_id)

    # Document-level co-occurrence: only among entities already found in that doc.
    for doc_id, present in doc_entity_keys.items():
        present_list = list(present)
        for i, a in enumerate(present_list):
            type_a = entities[a]["type"]
            for b in present_list[i + 1 : i + 6]:  # cap pairs per doc for speed
                type_b = entities[b]["type"]
                for dom, rng, rel in CO_OCCURRENCE_RELATIONS:
                    if type_a == dom and type_b == rng:
                        _add_triple(triples, graph, a, rel, b)
                    elif type_b == dom and type_a == rng:
                        _add_triple(triples, graph, b, rel, a)

    triple_list = list(triples)
    for subj, rel, obj in triple_list:
        graph.add_edge(subj, obj, relation=rel)

    if len(entities) < min_entities:
        raise RuntimeError(f"Built {len(entities)} entities; need at least {min_entities}.")
    if len(triple_list) < min_relationships:
        raise RuntimeError(f"Built {len(triple_list)} relationships; need at least {min_relationships}.")

    for name, meta in entities.items():
        meta["sources"] = sorted(source_map.get(name, []))

    return KnowledgeGraph(graph=graph, entities=entities, triples=triple_list)
