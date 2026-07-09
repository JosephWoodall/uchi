"""relational_reasoning.py -- deterministic transitive-relation veto layer
(0.4.0 follow-on to Item 17).

The gap this closes, demonstrated empirically before writing any code:
FactCheckOracle's word-overlap check treats a claim as a bag of terms, with
no notion of relational direction. Given evidence "A is taller than B" and
"B is taller than C", it accepted ALL THREE of: "A is taller than C"
(valid), "C is taller than A" (the reversed, logically wrong claim), and
"A is taller than A" (nonsense) -- identical term-overlap, zero ability to
tell them apart.

This does NOT attempt general logical-inference verification (that's a much
harder, open problem -- arbitrary causal/conditional reasoning, multi-step
arithmetic word problems, etc. are all explicitly out of scope). It handles
one narrow, well-defined, genuinely transitive class: simple comparative
relations (taller/shorter, older/younger, before/after, greater/less, and
the generic "more/less X than" pattern), extracted via cheap pattern
matching -- same "cheap heuristic, not full parsing" spirit as
iq_router.py's complexity regex, not a new parsing subsystem.

Two sources of evidence compose into the same graph: explicit comparative
sentences (extract_comparative), and bare numeric measurements on a shared
attribute (extract_attribute_value) -- real evidence is far more likely to
state "Building A is 442 meters tall" and "Building B is 330 meters tall"
separately than to state an explicit comparison, so relying only on
comparative wording would miss most real cases. Still narrow: catches only
the specific "is/was NUMBER [UNIT] ADJ" sentence shape, and only for the
attribute words in _ATTR_TO_RELATION.

Same additive-only safety invariant as every other layer in oracle.py:
this can only VETO a claim (when it contradicts the evidence's derivable
transitive closure, or is a trivial self-comparison), never independently
ACCEPT one the deterministic word-overlap check already rejected. When the
claim or evidence doesn't parse into a known comparative relation, or the
two entities aren't connected by any known chain, this abstains --
"no opinion", not a guess.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# comparative word -> (canonical relation name, direction)
# direction +1: subject > object on this relation. direction -1: subject < object.
_KNOWN_COMPARATIVES: dict[str, tuple[str, int]] = {
    "taller": ("height", 1), "shorter": ("height", -1),
    "older": ("age", 1), "younger": ("age", -1),
    "bigger": ("size", 1), "larger": ("size", 1), "smaller": ("size", -1),
    "faster": ("speed", 1), "slower": ("speed", -1),
    "heavier": ("weight", 1), "lighter": ("weight", -1),
    "greater": ("value", 1), "lesser": ("value", -1),
    "higher": ("height", 1), "lower": ("height", -1),
    "before": ("time", -1), "after": ("time", 1),
    "earlier": ("time", -1), "later": ("time", 1),
}

# "X is taller than Y" / "X happened before Y" style -- comparative word
# directly followed by "than", or a bare temporal comparative. Object stops
# at the first comma/terminal punctuation, not end-of-string -- a trailing
# clause ("..., based on the available data.") would otherwise get
# swallowed into the entity name, making "the black car" and "the black
# car, based on the available data" fail to match as the same node.
_COMPARATIVE_THAN_RE = re.compile(
    r"(?P<subj>.+?)\s+(?:is|was|are|were|happened|occurred)\s+"
    r"(?P<comp>\w+)\s+than\s+(?P<obj>[^,.!?]+)", re.I,
)
_TEMPORAL_RE = re.compile(
    r"(?P<subj>.+?)\s+(?:happened|occurred|came)\s+(?P<comp>before|after)\s+(?P<obj>[^,.!?]+)", re.I,
)
# generic "more/less ADJ than" -- e.g. "more expensive than", "less reliable than"
_MORE_LESS_RE = re.compile(
    r"(?P<subj>.+?)\s+(?:is|was|are|were)\s+(?P<sign>more|less)\s+(?P<attr>\w+)\s+than\s+(?P<obj>[^,.!?]+)", re.I,
)
# "Compared to Y, X is COMP." -- no "than" at all, a common real phrasing
# _COMPARATIVE_THAN_RE can't recognize (it requires the literal word
# "than"). Found missing via synthetic multi-hop data generation
# (verifier_train.py) using this exact phrasing to verify against this
# checker -- both fact-extraction calls silently returned None, which
# looked like a labeling bug in the generator until traced here.
_COMPARED_TO_RE = re.compile(
    r"compared\s+to\s+(?P<obj>[^,]+),\s+(?P<subj>.+?)\s+(?:is|was|are|were)\s+(?P<comp>\w+)(?=[,.!?]|$)", re.I,
)

# "X is NUMBER [UNIT] ATTR" -- e.g. "Building A is 442 meters tall.",
# "Alice is 30 years old." Real evidence states measurements this way far
# more often than it states explicit comparative sentences -- a graph built
# only from _COMPARATIVE_THAN_RE-style matches misses most real cases.
_VALUE_ATTR_RE = re.compile(
    r"(?P<subj>.+?)\s+(?:is|was|are|were|stands?|measures?)\s+"
    r"(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>[a-zA-Z]+)?\s+(?P<attr>[a-zA-Z]+)(?=[,.!?]|$)", re.I,
)
# attribute adjective -> canonical relation, same names _KNOWN_COMPARATIVES
# already uses so numeric-derived and comparative-sentence-derived edges
# compose into the same graph without any special-casing.
_ATTR_TO_RELATION: dict[str, str] = {
    "tall": "height", "high": "height", "old": "age", "big": "size", "large": "size",
    "heavy": "weight", "fast": "speed", "wide": "width", "long": "length",
    "deep": "depth", "expensive": "cost", "costly": "cost",
}


@dataclass
class ComparativeFact:
    subject: str
    relation: str
    obj: str
    direction: int  # +1: subject > obj, -1: subject < obj


def _norm(entity: str) -> str:
    return " ".join(entity.strip().lower().split())


def extract_comparative(sentence: str) -> Optional[ComparativeFact]:
    """Extract a single comparative fact from a sentence, or None if it
    doesn't match a known pattern -- graceful, not an error."""
    m = _MORE_LESS_RE.match(sentence.strip())
    if m:
        relation = m.group("attr").lower()
        direction = 1 if m.group("sign").lower() == "more" else -1
        return ComparativeFact(_norm(m.group("subj")), relation, _norm(m.group("obj")), direction)

    m = _TEMPORAL_RE.match(sentence.strip())
    if m:
        rel, base_dir = _KNOWN_COMPARATIVES[m.group("comp").lower()]
        return ComparativeFact(_norm(m.group("subj")), rel, _norm(m.group("obj")), base_dir)

    m = _COMPARATIVE_THAN_RE.match(sentence.strip())
    if m:
        comp = m.group("comp").lower()
        if comp not in _KNOWN_COMPARATIVES:
            return None
        rel, direction = _KNOWN_COMPARATIVES[comp]
        return ComparativeFact(_norm(m.group("subj")), rel, _norm(m.group("obj")), direction)

    m = _COMPARED_TO_RE.match(sentence.strip())
    if m:
        comp = m.group("comp").lower()
        if comp not in _KNOWN_COMPARATIVES:
            return None
        rel, direction = _KNOWN_COMPARATIVES[comp]
        return ComparativeFact(_norm(m.group("subj")), rel, _norm(m.group("obj")), direction)

    return None


def extract_attribute_value(sentence: str) -> Optional[tuple[str, str, float]]:
    """Extract (entity, canonical_relation, numeric_value) from a bare
    measurement sentence, or None if it doesn't match. Distinct from
    extract_comparative() -- this needs no comparative wording at all,
    just a stated number and a recognized attribute adjective."""
    m = _VALUE_ATTR_RE.match(sentence.strip())
    if not m:
        return None
    relation = _ATTR_TO_RELATION.get(m.group("attr").lower())
    if relation is None:
        return None
    try:
        value = float(m.group("value").replace(",", ""))
    except ValueError:
        return None
    return (_norm(m.group("subj")), relation, value)


class RelationalTransitivityChecker:
    """Builds a per-relation transitive closure from evidence's comparative
    facts, and vetoes candidate claims that contradict it or are trivially
    incoherent (self-comparison). Abstains -- never guesses -- whenever the
    claim or evidence doesn't parse, or the entities aren't connected by any
    known chain.
    """

    def _build_graphs(self, evidence: list[str]) -> dict[str, dict[str, set[str]]]:
        """relation -> {node: set(nodes reachable as "greater than" via a
        direct edge)} -- direction normalized so every edge means
        "greater on this relation", built before closure. Edges come from
        two sources, composed into the same graph: explicit comparative
        sentences, and pairs of bare numeric measurements on the same
        attribute -- evidence rarely states both facts about the same two
        entities as an explicit comparison, so relying only on the former
        misses most real cases."""
        edges: dict[str, dict[str, set[str]]] = {}

        def add_edge(relation: str, greater: str, lesser: str) -> None:
            graph = edges.setdefault(relation, {})
            graph.setdefault(greater, set()).add(lesser)
            graph.setdefault(lesser, set())

        for sentence in evidence:
            fact = extract_comparative(sentence)
            if fact is None:
                continue
            if fact.direction > 0:
                add_edge(fact.relation, fact.subject, fact.obj)
            else:
                add_edge(fact.relation, fact.obj, fact.subject)

        values_by_relation: dict[str, dict[str, float]] = {}
        for sentence in evidence:
            av = extract_attribute_value(sentence)
            if av is None:
                continue
            entity, relation, value = av
            values_by_relation.setdefault(relation, {})[entity] = value

        for relation, entity_values in values_by_relation.items():
            entities = list(entity_values.items())
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    e1, v1 = entities[i]
                    e2, v2 = entities[j]
                    if v1 > v2:
                        add_edge(relation, e1, e2)
                    elif v2 > v1:
                        add_edge(relation, e2, e1)
                    # equal values: no derivable direction, skip

        return edges

    def _transitive_closure(self, graph: dict[str, set[str]]) -> dict[str, set[str]]:
        """closure[x] = every node known to be "lesser than" x, direct or
        chained. Simple repeated-relaxation closure -- evidence sets here
        are small (a handful of retrieved passages), so this is cheap."""
        closure = {node: set(direct) for node, direct in graph.items()}
        changed = True
        while changed:
            changed = False
            for node in closure:
                new_reachable = set(closure[node])
                for lesser in closure[node]:
                    new_reachable |= closure.get(lesser, set())
                if new_reachable != closure[node]:
                    closure[node] = new_reachable
                    changed = True
        return closure

    def is_contradicted(self, claim: str, evidence: list[str]) -> bool:
        """True only when the claim is a well-formed comparative that
        contradicts the evidence's derivable transitive closure, or is a
        trivial self-comparison. False (abstain) whenever it can't
        confidently tell -- unparseable claim, unknown relation, or
        entities not connected by any known chain."""
        fact = extract_comparative(claim)
        if fact is None:
            return False
        if fact.subject == fact.obj:
            return True  # nothing is taller/older/etc. than itself

        graphs = self._build_graphs(evidence)
        graph = graphs.get(fact.relation)
        if graph is None:
            return False
        closure = self._transitive_closure(graph)

        greater, lesser = (fact.subject, fact.obj) if fact.direction > 0 else (fact.obj, fact.subject)
        if lesser in closure.get(greater, set()):
            return False  # consistent with (derivable from) the evidence
        if greater in closure.get(lesser, set()):
            return True  # the reverse relation is what's actually derivable -- contradiction
        return False  # not connected by any known chain -- genuinely unknown, not a guess
