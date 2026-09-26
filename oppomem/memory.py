"""Strategic-memory construction and the shared E5 / medoid / NLI retrieval path."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from .semantics import DirectionalNLI, E5EncoderLike, cosine
from .types import MemoryItem, MemoryMatch, TrainingRecord


_OPTION_CODE = re.compile(r"\b[ABCDE][1-5]\b")
_CARD_HINT_CODE = re.compile(r"\b(?:rank|card|hint)\s*[1-5]\b", re.I)


def valid_opponent_modeling(text: str, *, extra_forbidden: Sequence[str] = ()) -> bool:
    """Enforce the checkable parts of appendix C1 before a key enters retrieval.

    Semantic support and absence of hidden information still require the game
    adapter and the modeling prompt; they cannot be proven from a string alone.
    """

    clean = " ".join(text.split())
    if not clean or len(clean.split()) >= 60 or "insufficient evidence" in clean.casefold():
        return False
    for term in extra_forbidden:
        if term and re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", clean, re.I):
            return False
    return not (_OPTION_CODE.search(clean) or _CARD_HINT_CODE.search(clean))


def valid_response_strategy(text: str) -> bool:
    return bool(text.strip()) and len(text.split()) < 40


class StrategyReflector(Protocol):
    """Offline assistant; it receives only completed, player-visible training evidence."""

    def construct(self, game: str, canonical_modeling: str,
                  records: Sequence[TrainingRecord]) -> str: ...


@dataclass
class MemoryBank:
    game: str
    encoder_model: str
    nli_model: str
    items: dict[str, MemoryItem]
    batch_index: int = 0

    schema_version = 1

    def __post_init__(self) -> None:
        if not self.game or not self.encoder_model or not self.nli_model or self.batch_index < 0:
            raise ValueError("memory bank metadata is incomplete or invalid")
        if any(key != item.memory_id for key, item in self.items.items()):
            raise ValueError("memory bank keys must match item IDs")
        if any(not valid_opponent_modeling(item.opponent_modeling) or
               not valid_response_strategy(item.response_strategy)
               for item in self.items.values()):
            raise ValueError("memory bank contains an invalid key or response")

    def snapshot(self) -> "MemoryBank":
        """Detach the bank used during an interaction batch from later updates."""
        return MemoryBank(self.game, self.encoder_model, self.nli_model,
                          dict(self.items), self.batch_index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "game": self.game,
            "encoder_model": self.encoder_model,
            "nli_model": self.nli_model,
            "batch_index": self.batch_index,
            "items": [item.to_dict() for item in sorted(self.items.values(), key=lambda x: x.memory_id)],
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp",
                                                dir=destination.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2, allow_nan=False)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: str | Path) -> "MemoryBank":
        raw = json.loads(Path(path).read_text(encoding="utf-8"),
                         parse_constant=lambda value: (_ for _ in ()).throw(
                             ValueError(f"nonfinite bank value: {value}")
                         ))
        if raw.get("schema_version") != cls.schema_version:
            raise ValueError("unsupported OppoMem bank schema")
        items = [MemoryItem.from_dict(row) for row in raw["items"]]
        if len({item.memory_id for item in items}) != len(items):
            raise ValueError("duplicate memory IDs")
        return cls(str(raw["game"]), str(raw["encoder_model"]),
                   str(raw["nli_model"]), {item.memory_id: item for item in items},
                   int(raw.get("batch_index", 0)))


def _groups(vectors: np.ndarray, threshold: float) -> list[list[int]]:
    """Deterministic similarity grouping; the numeric threshold is run configuration."""
    parent = list(range(len(vectors)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if cosine(vectors[i], vectors[j]) >= threshold:
                a, b = find(i), find(j)
                parent[max(a, b)] = min(a, b)
    components: dict[int, list[int]] = {}
    for i in range(len(vectors)):
        components.setdefault(find(i), []).append(i)
    return [components[key] for key in sorted(components)]


def _medoid(group: list[int], vectors: np.ndarray) -> int:
    return max(group, key=lambda i: (
        sum(cosine(vectors[i], vectors[j]) for j in group), -i
    ))


class MemoryBuilder:
    """Create initial or newly discovered type-response pairs after training games."""

    def __init__(self, encoder: E5EncoderLike, nli: DirectionalNLI,
                 reflector: StrategyReflector, *, grouping_similarity: float):
        if not -1 <= grouping_similarity <= 1:
            raise ValueError("grouping_similarity must be in [-1, 1]")
        self.encoder, self.nli, self.reflector = encoder, nli, reflector
        self.grouping_similarity = float(grouping_similarity)

    def build(self, records: Sequence[TrainingRecord], *, game: str,
              previous: MemoryBank | None = None) -> tuple[MemoryBank, tuple[str, ...]]:
        """Use only training records; invalid C1 keys are excluded, not indexed."""
        if previous and previous.game != game:
            raise ValueError("cannot mix memory banks across games")
        if previous and (previous.encoder_model != self.encoder.model_name or
                         previous.nli_model != self.nli.model_name):
            raise ValueError("model versions differ from the frozen bank")
        ordered = sorted(records, key=lambda row: row.record_id)
        if len({row.record_id for row in ordered}) != len(ordered):
            raise ValueError("duplicate training record IDs")
        if any(row.game != game for row in ordered):
            raise ValueError("training records belong to a different game")
        accepted = [row for row in ordered if valid_opponent_modeling(
            row.opponent_modeling,
            extra_forbidden=(*row.forbidden_key_terms, *row.focal_actions),
        )]
        accepted_ids = {row.record_id for row in accepted}
        excluded = tuple(row.record_id for row in ordered if row.record_id not in accepted_ids)
        bank = previous.snapshot() if previous else MemoryBank(
            game, self.encoder.model_name, self.nli.model_name, {}
        )
        used_ids = {record_id for item in bank.items.values()
                    for record_id in item.supporting_record_ids}
        accepted = [row for row in accepted if row.record_id not in used_ids]
        if not accepted:
            return bank, excluded
        vectors = np.asarray(self.encoder.encode_passages(
            [row.opponent_modeling for row in accepted]
        ), dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(accepted):
            raise ValueError("encoder returned the wrong embedding shape")
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("encoder returned nonfinite or zero embeddings")
        for group in _groups(vectors, self.grouping_similarity):
            representative = _medoid(group, vectors)
            rows = [accepted[i] for i in group]
            key = accepted[representative].opponent_modeling
            vector = tuple(float(x) for x in vectors[representative])
            # New batch evidence can add a type; existing entries are revised by
            # utility evaluation, never silently overwritten during collection.
            if any(cosine(vector, item.medoid_vector) >= self.grouping_similarity
                   for item in bank.items.values()):
                continue
            response = self.reflector.construct(game, key, rows)
            if not valid_response_strategy(response):
                raise ValueError("reflection must return reusable advice under 40 words")
            record_ids = tuple(row.record_id for row in rows)
            memory_id = hashlib.sha256(json.dumps([game, key, record_ids],
                                                ensure_ascii=False).encode()).hexdigest()[:16]
            bank.items[memory_id] = MemoryItem(
                memory_id=memory_id,
                opponent_modeling=key,
                response_strategy=response.strip(),
                member_descriptions=tuple(row.opponent_modeling for row in rows),
                medoid_vector=vector,
                supporting_record_ids=record_ids,
            )
        return bank, excluded


class E5MedoidNLIRetriever:
    """E5 recall over actual-member medoids, then DeBERTa reranking to one item."""

    def __init__(self, encoder: E5EncoderLike, nli: DirectionalNLI,
                 *, candidate_count: int | None = None):
        if candidate_count is not None and (type(candidate_count) is not int or candidate_count < 1):
            raise ValueError("candidate_count must be positive")
        self.encoder, self.nli, self.candidate_count = encoder, nli, candidate_count

    def retrieve(self, opponent_modeling: str, bank: MemoryBank) -> MemoryMatch | None:
        if not bank.items:
            return None
        if bank.encoder_model != self.encoder.model_name or bank.nli_model != self.nli.model_name:
            raise ValueError("retrieval models differ from the bank's index")
        if not valid_opponent_modeling(opponent_modeling):
            return None
        query = np.asarray(self.encoder.encode_queries([opponent_modeling])[0], dtype=np.float32)
        recalled = sorted(
            ((cosine(query, item.medoid_vector), item) for item in bank.items.values()),
            key=lambda row: (-row[0], row[1].memory_id),
        )
        if self.candidate_count is not None:
            recalled = recalled[:self.candidate_count]
        scored = [
            MemoryMatch(item, similarity,
                        float(self.nli.entailment(opponent_modeling, item.opponent_modeling)))
            for similarity, item in recalled
        ]
        if any(not np.isfinite(row.nli_entailment) or not 0 <= row.nli_entailment <= 1
               for row in scored):
            raise ValueError("NLI entailment scores must be finite probabilities")
        return max(scored, key=lambda row: (
            row.nli_entailment, row.e5_cosine, row.item.memory_id
        ))
